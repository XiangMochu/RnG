import importlib
import os
import time
import wandb
import torch
from rich import print
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset
import torch.distributed as dist
from setup import init_config, init_distributed, init_wandb_and_backup
from utils.metric_utils import visualize_intermediate_results
from utils.training_utils import create_optimizer, create_lr_scheduler, auto_resume_job, print_rank0
from utils.metric_utils import export_results, summarize_evaluation
from tqdm import tqdm 


amp_dtype_mapping = {
    "fp16": torch.float16, 
    "bf16": torch.bfloat16, 
    "fp32": torch.float32, 
    'tf32': torch.float32
}

class Trainer:
    def __init__(self, config):
        self.config = config
        self._init_data()
        self._init_model()

    def _init_data(self):
        config = self.config

        dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
        module, class_name = dataset_name.rsplit(".", 1)
        Dataset = importlib.import_module(module).__dict__[class_name]
        dataset = Dataset(config)

        if hasattr(config.training, 'dataset_name2'):
            dataset_name2 = config.training.get("dataset_name2")
            module, class_name = dataset_name2.rsplit(".", 1)
            Dataset2 = importlib.import_module(module).__dict__[class_name]
            dataset2 = Dataset2(config, is_second=True)

            dataset = ConcatDataset([dataset, dataset2])

        batch_size_per_gpu = config.training.batch_size_per_gpu

        datasampler = DistributedSampler(dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size_per_gpu,
            shuffle=False,
            num_workers=config.training.num_workers,
            persistent_workers=True,
            pin_memory=False,
            drop_last=True,
            prefetch_factor=config.training.prefetch_factor,
            sampler=datasampler,
        )
        self.datasampler = datasampler
        self.dataloader_iter = iter(dataloader)

        # Validation dataset
        val_dataset_name = config.training.get("val_dataset_name")
        module, class_name = val_dataset_name.rsplit(".", 1)
        ValDataset = importlib.import_module(module).__dict__[class_name]
        val_dataset = ValDataset(config)

        val_datasampler = DistributedSampler(val_dataset)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.training.val_dataset_cfgs.training.batch_size_per_gpu,
            shuffle=False,
            num_workers=config.training.num_workers,
            persistent_workers=True,
            pin_memory=False,
            drop_last=True,
            prefetch_factor=config.training.prefetch_factor,
            sampler=val_datasampler,
        )
        self.val_dataloader = val_dataloader

        total_train_steps = config.training.train_steps
        grad_accum_steps = config.training.grad_accum_steps
        total_param_update_steps = total_train_steps
        total_train_steps = total_train_steps * grad_accum_steps # real train steps when using gradient accumulation
        total_batch_size = batch_size_per_gpu * config.ddp_info.world_size * grad_accum_steps
        total_num_epochs = int(total_param_update_steps * total_batch_size / len(dataset))

        self.dataset = dataset
        self.dataloader = dataloader
        self.total_train_steps = total_train_steps
        self.total_param_update_steps = total_param_update_steps
        self.total_batch_size = total_batch_size
        self.total_num_epochs = total_num_epochs
        self.grad_accum_steps = grad_accum_steps

    def _init_model(self):
        config = self.config
        ddp_info = config.ddp_info

        module, class_name = config.model.class_name.rsplit(".", 1)
        LVSM = importlib.import_module(module).__dict__[class_name]
        model = LVSM(config).to(ddp_info.device)

        if config.training.get('use_bf16', False):
            model = model.to(amp_dtype_mapping['bf16'])
            # all DPT heads use fp32 instead, won't converge with bf16
            model.camera_head.to(torch.float32)
            model.point_head.to(torch.float32)
            model.rgb_head.to(torch.float32)
            model.loss_computer.to(torch.float32)
            print_rank0("Using bf16 training!")

        model = DDP(model, device_ids=[ddp_info.local_rank])

        optimizer, optimized_param_dict, all_param_dict = create_optimizer(
            model,
            config.training.weight_decay,
            config.training.lr,
            (config.training.beta1, config.training.beta2),
        )
        optim_param_list = list(optimized_param_dict.values())

        scheduler_type = config.training.get("scheduler_type", "cosine")
        lr_scheduler = create_lr_scheduler(
            optimizer,
            self.total_param_update_steps,
            config.training.warmup,
            scheduler_type=scheduler_type,
        )

        if config.training.get("resume_ckpt", "") != "":
            ckpt_load_path = config.training.resume_ckpt
        else:
            ckpt_load_path = config.training.checkpoint_dir
        reset_training_state = config.training.get("reset_training_state", False)
        optimizer, lr_scheduler, cur_train_step, cur_param_update_step = auto_resume_job(
            ckpt_load_path,
            model,
            optimizer,
            lr_scheduler,
            reset_training_state,
        )

        enable_grad_scaler = config.training.use_amp and config.training.amp_dtype == "fp16"
        self.scaler = torch.amp.GradScaler('cuda', enabled=enable_grad_scaler)
        print_rank0(f"Grad scaler enabled: {enable_grad_scaler}")
        dist.barrier()

        self.model = model
        self.optimized_param_dict = optimized_param_dict
        self.optim_param_list = optim_param_list
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.cur_train_step = cur_train_step
        self.cur_param_update_step = cur_param_update_step
    
    def fetch_data1(self, cur_epoch):
        config = self.config
        ddp_info = config.ddp_info

        try:
            data = next(self.dataloader_iter)
        except StopIteration:
            print(f"Current Rank {ddp_info.local_rank} Ran out of data. Resetting dataloader epoch to {cur_epoch}; might take a while...")
            self.datasampler.set_epoch(cur_epoch)
            self.dataloader_iter = iter(self.dataloader)
            data = next(self.dataloader_iter)
        return data

    def run(self):
        config = self.config
        ddp_info = config.ddp_info

        self.start_train_step = self.cur_train_step
        self.model.train()

        while self.cur_train_step <= self.total_train_steps:
            tic = time.time()
            cur_epoch = int(self.cur_train_step * (self.total_batch_size / self.grad_accum_steps) // len(self.dataset) )
            
            ### get data
            data = self.fetch_data1(cur_epoch)

            batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in data.items()}

            if config.training.get('use_bf16', False):
                batch = {k: v.to(amp_dtype_mapping['bf16']) if type(v) == torch.Tensor else v for k, v in batch.items()}
                # ret_dict = self.model(batch, exclude_bg=self.cur_train_step<self.total_train_steps//4)

            # else:
            with torch.autocast(
                enabled=config.training.use_amp,
                device_type="cuda",
                dtype=amp_dtype_mapping[config.training.amp_dtype],
            ):
                ret_dict = self.model(batch, exclude_bg=self.cur_train_step<self.total_train_steps//4)
            
            update_grads = (self.cur_train_step + 1) % self.grad_accum_steps == 0 or self.cur_train_step == self.total_train_steps
            if update_grads:
                with self.model.no_sync(): # no sync grads for efficiency
                    self.scaler.scale(ret_dict.loss_metrics.loss / self.grad_accum_steps).backward()
            else:
                self.scaler.scale(ret_dict.loss_metrics.loss / self.grad_accum_steps).backward()
            self.cur_train_step += 1

            export_inter_results = ((self.cur_train_step-1) == self.start_train_step) or (self.cur_train_step % config.training.vis_every == 0)

            skip_optimizer_step = False
            # Skip optimizer step if loss is NaN or Inf
            if torch.isnan(ret_dict.loss_metrics.loss) or torch.isinf(ret_dict.loss_metrics.loss):
                print(f"NaN or Inf loss detected, skip this iteration")
                skip_optimizer_step = True
                ret_dict.loss_metrics.loss.data = torch.zeros_like(ret_dict.loss_metrics.loss)

            total_grad_norm = None
            # Check gradient norm and update optimizer if everything is fine
            if update_grads and (not skip_optimizer_step):
                # Unscales the gradients
                self.scaler.unscale_(self.optimizer) 
                # For all gradients, we safely change the NaN -> 0., inf -> 1e-6, -inf -> 1e-6.
                with torch.no_grad():
                    for n, p in self.optimized_param_dict.items():
                        if p.requires_grad and (p.grad is not None):
                            p.grad.nan_to_num_(nan=0.0, posinf=1e-6, neginf=-1e-6)
            
                # visualize the grad norm of each layer of our transformer (FOR DEBUG)
                if ddp_info.is_main_process and config.training.get("log_grad_norm_details", False):
                    grad_norms = {}  # Dictionary to store norms per layer
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:  # Some parameters might not have gradients
                            grad_norms[name] = param.grad.detach().norm().item()  # Detach for safety
                    for layer_name, grad_norm in grad_norms.items():
                        wandb.log({"grad_norm_details/" + layer_name: grad_norm}, step=self.cur_train_step)

                total_grad_norm = 0.0
                if config.training.grad_clip_norm > 0:
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(self.optim_param_list, max_norm=config.training.grad_clip_norm).item()

                    if total_grad_norm > config.training.grad_clip_norm * 2.0:
                        print(f"WARNING: step {self.cur_train_step} grad norm too large {total_grad_norm} > {config.training.grad_clip_norm * 2.0}")

                    allowed_gradnorm = config.training.grad_clip_norm * config.training.get("allowed_gradnorm_factor", 5)
                    if total_grad_norm > allowed_gradnorm:
                        skip_optimizer_step = True
                        print(f"WARNING: step {self.cur_train_step} grad norm too large {total_grad_norm} > {allowed_gradnorm}, skipping optimizer step")

                    # show grad norm in wandb if it's too large
                    display_grad_norm = total_grad_norm > config.training.grad_clip_norm * 2.0 or total_grad_norm > allowed_gradnorm
                    if display_grad_norm and ddp_info.is_main_process:
                        wandb.log({"grad_norm": total_grad_norm}, step=self.cur_train_step)

                if not skip_optimizer_step:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.cur_param_update_step += 1

                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            # log and save checkpoint
            if ddp_info.is_main_process:
                self.log_and_save_ckpt(ret_dict, cur_epoch, self.cur_param_update_step, tic, export_inter_results, total_grad_norm)

            if (self.cur_train_step % config.training.checkpoint_every == 0) or (self.cur_train_step == self.total_train_steps):
                self.validate()

            if export_inter_results:
                torch.cuda.empty_cache()
                dist.barrier()

    @torch.inference_mode()
    def validate(self):
        config = self.config
        ddp_info = config.ddp_info
        self.model.eval()

        for batch in tqdm(self.val_dataloader, disable = not ddp_info.is_main_process):
            batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in batch.items()}

            with torch.autocast(enabled=config.training.use_amp, device_type="cuda",
                    dtype=amp_dtype_mapping[config.training.amp_dtype]):
                ret_dict = self.model(batch, target_has_input=False, is_valid=True)
            
            out_dir = os.path.join(config.training.validation_out_dir, f"eval_iter_{self.cur_train_step:08d}")
            export_results(ret_dict, out_dir, compute_metrics=True)

        if ddp_info.is_main_process:
            avg_metric_dict = summarize_evaluation(out_dir, ret_dict=True)
            # print in console
            print(f"Validation summary at step {self.cur_train_step}: ")
            for k, v in avg_metric_dict.items():
                print(f"{k}: {v}")

            # log to wandb
            wandb.log({"val/" + k: float(v) for k, v in avg_metric_dict.items()}, step=self.cur_train_step)
        
        dist.barrier()
        self.model.train()

    def log_and_save_ckpt(self, ret_dict, cur_epoch, cur_param_update_step, tic, export_inter_results, total_grad_norm):
        config = self.config

        # loss_dict = {k: float(f"{v.item():.6f}") for k, v in ret_dict.loss_metrics.items()}
        loss_dict = {}
        for k,v in ret_dict.loss_metrics.items():
            if isinstance(v, torch.Tensor):
                loss_dict[k] = float(f"{v.item():.6f}")
            elif isinstance(v, float):
                loss_dict[k] = v
            elif isinstance(v, int):
                loss_dict[k] = float(v)
            else:
                raise ValueError(f"Unknown type of loss value {type(v)}")

        # print in console
        if (self.cur_train_step % config.training.print_every == 0) or (self.cur_train_step < 100 + self.start_train_step):
            print_str = f"[Epoch {int(cur_epoch):>3d}] | Forwad step: {int(self.cur_train_step):>6d} (Param update step: {int(cur_param_update_step):>6d})"
            print_str += f" | Iter Time: {time.time() - tic:.2f}s | LR: {self.optimizer.param_groups[0]['lr']:.6f}\n"
            # Add loss values
            for k, v in loss_dict.items():
                print_str += f"{k}: {v:.6f} | "
            print(print_str)

        # log in wandb
        if (self.cur_train_step % config.training.wandb_log_every == 0) or (
            self.cur_train_step < 200 + self.start_train_step
        ):
            log_dict = {
                "iter": self.cur_train_step, 
                "forward_pass_step": self.cur_train_step,
                "param_update_step": cur_param_update_step,
                "lr": self.optimizer.param_groups[0]["lr"],
                "iter_time": time.time() - tic,
                "grad_norm": total_grad_norm,
                "epoch": cur_epoch,
            }
            log_dict.update({"train/" + k: v for k, v in loss_dict.items()})
            wandb.log(
                log_dict,
                step=self.cur_train_step,
            )

        # save checkpoint
        if (self.cur_train_step % config.training.checkpoint_every == 0) or (self.cur_train_step == self.total_train_steps):
            if isinstance(self.model, DDP):
                model_weights = self.model.module.state_dict()
            else:
                model_weights = self.model.state_dict()
            checkpoint = {
                "model": model_weights,
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "fwdbwd_pass_step": self.cur_train_step,
                "param_update_step": cur_param_update_step,
            }
            os.makedirs(config.training.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(config.training.checkpoint_dir, f"ckpt_{self.cur_train_step:016}.pt")
            torch.save(checkpoint, ckpt_path)
            print(f"Saved checkpoint at step {self.cur_train_step} to {os.path.abspath(ckpt_path)}")
        
        # export intermediate visualization results
        if export_inter_results:
            vis_path = os.path.join(config.training.checkpoint_dir, f"iter_{self.cur_train_step:08d}")
            os.makedirs(vis_path, exist_ok=True)
            visualize_intermediate_results(vis_path, ret_dict)
            torch.cuda.empty_cache()
            self.model.train()

                    
if __name__ == '__main__':
    # Load config and read(override) arguments from CLI
    config = init_config()

    os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

    # Set up DDP for training/inference and Fix random seed
    ddp_info = init_distributed(seed=777)
    dist.barrier()

    # Set up wandb and backup source code
    if ddp_info.is_main_process:
        init_wandb_and_backup(config)
    dist.barrier()

    # Set up tf32
    torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
    torch.backends.cudnn.allow_tf32 = config.training.use_tf32

    # Start training
    config.ddp_info = ddp_info
    trainer = Trainer(config)
    trainer.run()

    dist.barrier()
    dist.destroy_process_group()
