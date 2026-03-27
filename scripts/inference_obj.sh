CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 8 --nnodes 1 \
 --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29509 \
 inference.py --config "configs/RnG_obj_small_bf16_15k.yaml" \
 inference_out_dir = ./experiments/evaluation/RnG \
 training.checkpoint_dir = ./experiments/checkpoints/RnG_Med \
 training.target_has_input =  false \
 training.val_dataset_cfgs.split_file = 'data/gso.txt' \
 training.num_views = 14 \
 training.num_input_views = 4 \
 training.num_target_views = 10 \
