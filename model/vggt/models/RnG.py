# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from einops import rearrange, repeat
import torch
import traceback
import os
import math
import torch.nn as nn
from easydict import EasyDict as edict

from model.vggt.models.aggregator import Aggregator, Aggregator_with_kv_cache
from model.vggt.heads.camera_head import CameraHead
from model.vggt.heads.dpt_head import DPTHead
from einops.layers.torch import Rearrange
from utils import data_utils 
from model.vggt.layers.vision_transformer import init_weights_vit_timm as init_weights
from model.loss import MultiTaskLossComputer


class RnG(nn.Module):
    def __init__(self, config, embed_dim=1024, **kwargs):
        super().__init__()
        self.config = config
        img_size = self.config.model.image_tokenizer.image_size
        patch_size = self.config.model.image_tokenizer.patch_size
        self.img_size = img_size
        self.patch_size = patch_size

        self.embed_dim = embed_dim
        self.kwargs = kwargs

        if kwargs.get('use_kv_cache', False):
            self.aggregator = Aggregator_with_kv_cache(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        else:
            self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, patch_size=patch_size, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, patch_size=patch_size, output_dim=2, activation="exp", conf_activation="expp1")

        self.process_data = data_utils.ProcessData(config)
        self.process_val_data = data_utils.ProcessData(config.training.val_dataset_cfgs)
        self.make_modifications()
        # self.loss_computer = MultiTaskLossComputer(config)

    def make_modifications(self):
        if self.kwargs.get('is_debugging', False) == False:
            if hasattr(self.config.model, 'pretrained_path'):
                print("Loading pretrained weights from ", self.config.model.pretrained_path)
                checkpoint = torch.load(self.config.model.pretrained_path, map_location="cpu")

                patch_embed_weight = checkpoint['aggregator.patch_embed.patch_embed.proj.weight']
                if patch_embed_weight.shape[-1] != self.patch_size:
                    # resize patch embedding weight
                    new_patch_embed_weight = torch.nn.functional.interpolate(
                        patch_embed_weight, size=(self.patch_size, self.patch_size), mode='bilinear')
                    checkpoint['aggregator.patch_embed.patch_embed.proj.weight'] = new_patch_embed_weight
                    print(f'Resizing patch embedding weight from {patch_embed_weight.shape} to {new_patch_embed_weight.shape}.')

                    # resize pose embedding weight
                    pose_embed = checkpoint['aggregator.patch_embed.pos_embed']
                    cls_embed = pose_embed[:, :1, :]
                    img_embed = pose_embed[:, 1:, :]
                    hw = img_embed.shape[1]
                    h = int(math.sqrt(hw))
                    img_embed = rearrange(img_embed, 'b (h w) c -> b c h w', h=h)
                    new_h = self.img_size // self.patch_size
                    new_img_embed = torch.nn.functional.interpolate(
                        img_embed, size=(new_h, new_h), mode='bilinear')
                    new_img_embed = rearrange(new_img_embed, 'b c h w -> b (h w) c')
                    new_pose_embed = torch.cat([cls_embed, new_img_embed], dim=1)
                    checkpoint['aggregator.patch_embed.pos_embed'] = new_pose_embed
                    print(f'Resizing pose embedding weight from {pose_embed.shape} to {new_pose_embed.shape}.')

                checkpoint = {k:v for k,v in checkpoint.items() if not k.startswith('track_head')}
                self.load_state_dict(checkpoint, strict=True)
                print("Loaded pretrained weights successfully.")
        else:
            print("Debug mode enabled. Not loading any pretrained weights.")
        self.modify_heads()
        self.add_pose_tokenizer()
        self.freeze_dino()

    def modify_heads(self):
        del self.depth_head
        self.rgb_head = deepcopy(self.point_head)
        self.rgb_head.activation = "sigmoid"
        self.rgb_head.scratch.output_conv1.apply(init_weights)
        self.rgb_head.scratch.output_conv2.apply(init_weights)

    def add_pose_tokenizer(self):
        self.pose_tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> b v (hh ww) (ph pw c)",
                ph=self.patch_size,
                pw=self.patch_size),
            nn.Linear(
                6 * (self.patch_size**2),
                self.embed_dim,
                bias=False))
        self.pose_tokenizer.apply(init_weights)

    def freeze_dino(self):
        print("Freezing DINO weights...")
        for param in self.aggregator.patch_embed.parameters():
            param.requires_grad = False

    def get_posed_input(self, images=None, ray_o=None, ray_d=None, method="default_plucker"):
        if method == "custom_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            pose_cond = torch.cat([ray_d, nearest_pts], dim=2)
            
        elif method == "aug_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d, nearest_pts], dim=2)
            
        else:  # default_plucker
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d], dim=2)

        if images is None:
            return pose_cond
        else:
            return images, pose_cond

    def _forward(self, images: torch.Tensor):
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}
        with torch.cuda.amp.autocast(enabled=False):
            rgb, conf = self.rgb_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx)
            predictions["rgb"] = rgb
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions

    def forward(self, data_batch, has_target_image=True, target_has_input=None, 
                exclude_bg=False, is_valid=False):
        if target_has_input is None:
            target_has_input = self.config.training.target_has_input

        if is_valid:
            process_data = self.process_val_data
        else:
            process_data = self.process_data

        input, target = process_data(
            data_batch, 
            has_target_image=has_target_image, 
            target_has_input=target_has_input, 
            compute_rays=True)

        # Process input images
        input_pose_cond = self.get_posed_input(ray_o=input.ray_o, ray_d=input.ray_d)
        b, v_input, c, h, w = input_pose_cond.size()

        # Process target pose
        target_pose_cond = self.get_posed_input(ray_o=target.ray_o, ray_d=target.ray_d)
        b, v_target, c, h, w = target_pose_cond.size()

        # b, (v_input + v_target), c, h, w
        pose_cond = torch.concat([input_pose_cond, target_pose_cond], dim=1)
        # (b v) n c
        pose_tokens = self.pose_tokenizer(pose_cond)
        
        aggregated_tokens_list, patch_start_idx = self.aggregator(input.image, pose_tokens, posed_input=not self.config.unposed)

        aggregated_tokens_list = [i.float() for i in aggregated_tokens_list]
        target_pose_cond = target_pose_cond.float()

        # discard input view tokens and camera/register tokens.
        # because each feat has dimension [b, v, ...],
        # the v dimension is [i, ..., o]: only the last one is the output
        target_views_tokens_list = [feat[:,-1:] for feat in aggregated_tokens_list]

        input_views_tokens_list = [feat[:, :-1] for feat in aggregated_tokens_list]

        with torch.cuda.amp.autocast(enabled=False):
            rendered_images, conf = self.rgb_head(
                target_views_tokens_list, images=target_pose_cond, patch_start_idx=patch_start_idx)
            rendered_images = rearrange(rendered_images, 'b v h w c -> b v c h w')

            target_points, pts_conf = self.point_head(
                target_views_tokens_list, images=target_pose_cond, patch_start_idx=patch_start_idx)
            target_points = rearrange(target_points, 'b v h w c -> b v c h w', c=3)

            pose_enc_list = self.camera_head(input_views_tokens_list)

        pts_conf = rearrange(pts_conf, 'b v h w -> b v 1 h w')

        # print(input.extrinsic[:,0]) # must be [I_3x3, [0,0,1]^T], or [I_3x3, [0,0,-1]^T]^{-1}

        input_extrinsic = repeat(input.extrinsic, 'b v_in i j -> (b v_out) v_in i j', v_out=v_target)
        input_intrinsic = repeat(input.intrinsic, 'b v_in i j -> (b v_out) v_in i j', v_out=v_target)

        if has_target_image:
            loss_metrics = self.loss_computer(
                rendering=rendered_images,
                target=target.image,
                exclude_bg=exclude_bg,
                pts_est=target_points,
                pts_conf=pts_conf,
                pts_gt = getattr(target, 'point_map', None),
                pose_enc_list=pose_enc_list,
                extrinsics=input_extrinsic,
                intrinsics=input_intrinsic,
                image_hw=(h, w)
            )
        else:
            loss_metrics = None

        result = edict(
            input=input,
            target=target,
            loss_metrics=loss_metrics,
            render=rendered_images,
            points=target_points,
            camera=pose_enc_list
            )
        
        return result

    def forward_single_target_view_unposed(self, input_batch, target_c2w):
        input, target = self.process_data.forward_single_target_view(
            input_batch, target_c2w)

        target_pose_cond = self.get_posed_input(ray_o=target.ray_o, ray_d=target.ray_d)
        b, v_target, c, h, w = target_pose_cond.size()

        b, v_input, *_ = input.image.shape
        input_pose_cond = torch.zeros(b, v_input, c, h, w, device=target_pose_cond.device, dtype=target_pose_cond.dtype)

        pose_cond = torch.concat([input_pose_cond, target_pose_cond], dim=1)
        pose_cond = self.pose_tokenizer(pose_cond)

        aggregated_tokens_list, patch_start_idx = self.aggregator(input.image, pose_cond, posed_input=False)

        aggregated_tokens_list = [i.float() for i in aggregated_tokens_list]
        target_pose_cond = target_pose_cond.float()

        # discard input view tokens and camera/register tokens.
        # because each feat has dimension [b, v, ...],
        # the v dimension is [i, ..., o]: only the last one is the output
        target_views_tokens_list = [feat[:,-1:] for feat in aggregated_tokens_list]

        input_views_tokens_list = [feat[:, :-1] for feat in aggregated_tokens_list]

        with torch.cuda.amp.autocast(enabled=False):
            rendered_images, conf = self.rgb_head(
                target_views_tokens_list, images=target_pose_cond, patch_start_idx=patch_start_idx)
            rendered_images = rearrange(rendered_images, 'b v h w c -> b v c h w')

            target_points, pts_conf = self.point_head(
                target_views_tokens_list, images=target_pose_cond, patch_start_idx=patch_start_idx)
            target_points = rearrange(target_points, 'b v h w c -> b v c h w', c=3)

            pose_enc_list = self.camera_head(input_views_tokens_list)

        pts_conf = rearrange(pts_conf, 'b v h w -> b v 1 h w')

        result = edict(
            input=input,
            target=target,
            render=rendered_images,
            points=target_points,
            points_conf=pts_conf,
            camera=pose_enc_list)

        return result


    def forward_pose_only(self, input_images):
        ### disable all attention mask
        for block in self.aggregator.global_blocks:
            block.attn.mask_attention = None
            block.attn.mode = 'save_cache'

        aggregated_tokens_list, patch_start_idx = self.aggregator.forward_pose_only(input_images)
        with torch.cuda.amp.autocast(enabled=False):
            pose_enc_list = self.camera_head(aggregated_tokens_list)

        ### recover all attention mask
        for block in self.aggregator.global_blocks:
            block.attn.mask_attention = 1029

        return pose_enc_list[-1]
        
    def forward_rendering_using_kv_cache(self, target_c2w):
        for block in self.aggregator.global_blocks:
            block.attn.mode = 'read_cache'

        target = self.process_data.forward_target_view_without_input(target_c2w)
        target_pose_cond = self.get_posed_input(ray_o=target.ray_o, ray_d=target.ray_d)
        b, v_target, c, h, w = target_pose_cond.size()

        pose_cond = target_pose_cond
        pose_cond = self.pose_tokenizer(pose_cond)

        aggregated_tokens_list, patch_start_idx = self.aggregator.forward_target_view_reading_kv_cache(pose_cond)

        aggregated_tokens_list = [i.float() for i in aggregated_tokens_list]
        target_pose_cond = target_pose_cond.float()

        target_views_tokens_list = aggregated_tokens_list

        # with torch.cuda.amp.autocast(enabled=False):
        with torch.amp.autocast(enabled=False, device_type='cuda'):
            rendered_images, conf = self.rgb_head(
                target_views_tokens_list, images=target_pose_cond, patch_start_idx=patch_start_idx)
            rendered_images = rearrange(rendered_images, 'b v h w c -> b v c h w')

            target_points, pts_conf = self.point_head(
                target_views_tokens_list, images=target_pose_cond, patch_start_idx=patch_start_idx)
            target_points = rearrange(target_points, 'b v h w c -> b v c h w', c=3)

        pts_conf = rearrange(pts_conf, 'b v h w -> b v 1 h w')

        result = edict(
            target=target,
            render=rendered_images,
            points=target_points,
            points_conf=pts_conf)
        
        # for block in self.aggregator.global_blocks:
        #     block.attn.mode = 'save_cache'

        return result

    @torch.no_grad()
    def load_ckpt(self, load_path):
        if os.path.isdir(load_path):
            ckpt_names = [file_name for file_name in os.listdir(load_path) if file_name.endswith(".pt")]
            ckpt_names = sorted(ckpt_names, key=lambda x: x)
            ckpt_paths = [os.path.join(load_path, ckpt_name) for ckpt_name in ckpt_names]
        else:
            ckpt_paths = [load_path]
        try:
            checkpoint = torch.load(ckpt_paths[-1], map_location="cpu", weights_only=True)
        except:
            traceback.print_exc()
            print(f"Failed to load {ckpt_paths[-1]}")
            return None
        
        self.load_state_dict(checkpoint["model"], strict=False)
        return 0