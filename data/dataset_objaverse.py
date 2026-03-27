import random
import traceback
import os
import math
from einops import rearrange
import numpy as np
import PIL
import torch
from torch.utils.data import Dataset
import json
import torch.nn.functional as F
from PIL import Image
from viser import transforms as vtf
import imageio


class ObjaverseDataset(Dataset):
    def __init__(self, config, is_second=False):
        self.config = config
        if is_second:
            self.root_path = self.config.training.root_path2
            data_path = self.config.training.dataset_path2
            self.total_frames_per_obj = self.config.training.total_frames_per_obj2
        else:
            self.root_path = self.config.training.root_path
            data_path = self.config.training.dataset_path
            self.total_frames_per_obj = self.config.training.total_frames_per_obj
        try:
            with open(data_path, 'r') as f:
                self.all_object_list = f.readlines()
            self.all_object_list = [l.strip() for l in self.all_object_list]
            
        except Exception as e:
            print(f"Error reading dataset paths from '{data_path}'")
            raise e
        
        self.fov = 0.6981317007977318

    def __len__(self):
        return len(self.all_object_list)

    def view_selector(self):
        if self.total_frames_per_obj < self.config.training.num_views:
            raise ValueError(f"total_frames_per_obj ({self.total_frames_per_obj}) is smaller than num_views ({self.config.training.num_views})")
        
        sampled_frames = random.sample(range(0, self.total_frames_per_obj), self.config.training.num_views)
        return sampled_frames

    @staticmethod
    def transform_pose(pose):
        pose = np.array(pose)
        rpy = vtf.SO3.from_matrix(pose[:3,:3]).as_rpy_radians()
        r,p,y = rpy.roll, rpy.pitch, rpy.yaw
        rpy = [r-np.pi/2, -y, p]
        
        pos = pose[:3,3]
        x,y,z = pos
        xyz = [x, -z, y]

        rot = vtf.SO3.from_rpy_radians(*rpy)
        wxyz = rot.wxyz
        position = np.array(xyz)

        pose1 = np.eye(4)
        pose1[:3,:3] = rot.as_matrix()
        pose1[:3,3] = position
        return pose1

    @staticmethod
    def pinhole_z_depth_to_xyz(depth, f, H=512, W=512):
        if isinstance(depth, torch.Tensor):
            depth = depth.numpy()
        if not isinstance(depth, float):
            H, W = depth.shape
            z = depth
        else:
            z = np.ones((H, W), dtype=np.float32) * depth
        y, x = np.mgrid[:H, :W]
        x = x - W // 2
        y = H // 2 - y
        x = x / f * z
        y = - y / f * z
        return np.stack([x, y, z], -1)

    def preprocess_frames(self, object_name, image_indices):
        resize_h = self.config.model.image_tokenizer.image_size
        patch_size = self.config.model.image_tokenizer.patch_size

        images = []
        fxfycxcys = []
        intrinsics = []
        c2ws = []
        point_maps = []
        depth_maps = []

        all_pose_path = os.path.join(self.root_path, object_name, f'transforms.json')
        with open(all_pose_path, 'r') as f:
            all_poses = json.load(f)['frames']

        target_first_view = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,-1],[0,0,0,1.]])

        for v_idx, img_idx in enumerate(image_indices):
            ### image
            cur_image_path = os.path.join(self.root_path, object_name, f'{img_idx:03d}.png')
            image = Image.open(cur_image_path)
            white_bg = Image.new(mode='RGBA', size=image.size, color=(255,)*4)
            image = Image.alpha_composite(white_bg, image)
            image = image.convert("RGB")

            original_image_w, original_image_h = image.size
            
            resize_w = int(resize_h / original_image_h * original_image_w)
            resize_w = int(round(resize_w / patch_size) * patch_size)
            image = image.resize((resize_w, resize_h), resample=Image.LANCZOS)

            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()

            ### intrinsic
            focal_length = (resize_w/2) / math.tan(self.fov/2)
            fxfycxcy = torch.tensor([focal_length, focal_length, resize_w/2, resize_h/2])
            intrinsic_mat = torch.tensor([[focal_length, 0, resize_w/2], [0., focal_length, resize_h/2], [0., 0., 1]])
            images.append(image)
            fxfycxcys.append(fxfycxcy)
            intrinsics.append(intrinsic_mat)


            ### extrinsic
            pose = all_poses[img_idx]['transform_matrix']
            c2w = torch.from_numpy(self.transform_pose(pose)).float()
            if v_idx == 0:
                norm_first_view_t = torch.norm(c2w[:3,3])
                c2w[:3,3] /= norm_first_view_t
                inv_first_view = torch.inverse(c2w)
            else:
                c2w[:3,3] /= norm_first_view_t
            c2w = target_first_view @ inv_first_view @ c2w

            c2ws.append(c2w)

            ### depth & point map
            cur_depth_path = os.path.join(self.root_path, object_name, f'{img_idx:03d}_depth.png')
            depth = Image.open(cur_depth_path)
            depth = np.array(depth.resize((resize_w, resize_h), resample=Image.NEAREST))
            mask = depth < 65534 # valid mask, 1=fg, 0=bg

            max_depth = all_poses[img_idx]['depth']['max']
            min_depth = all_poses[img_idx]['depth']['min']
            depth_range = max_depth - min_depth

            depth = depth / 65535.0 * depth_range + min_depth
            depth = depth * mask

            # normalize by first view's translation
            depth = torch.from_numpy(depth) / norm_first_view_t

            depth_maps.append(depth.float())

            pt = self.pinhole_z_depth_to_xyz(depth, focal_length)
            pt = torch.from_numpy(pt).float()

            # transform to world coordinate
            h,w,c = pt.shape
            pt = rearrange(pt, 'h w c -> (h w) c')
            ones = torch.ones((h*w, 1), dtype=pt.dtype)
            pt = torch.cat([pt, ones], dim=-1)
            pt = c2w @ pt.T
            pt = pt[:3].T
            pt = rearrange(pt, '(h w) c -> c h w', h=h, w=w)
            pt = pt * torch.from_numpy(mask)[None,:,:].float()

            point_maps.append(pt)

        images = torch.stack(images, dim=0)
        fxfycxcys = torch.stack(fxfycxcys, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)
        c2ws = torch.stack(c2ws)
        depth_maps = torch.stack(depth_maps, dim=0)
        point_maps = torch.stack(point_maps, dim=0)
        
        return images, fxfycxcys, intrinsics, c2ws, point_maps, depth_maps
        

    def __getitem__(self, idx):
        object_name = self.all_object_list[idx]
        image_indices = self.view_selector()
        input_images, input_fxfycxcy, input_intrinsics, input_c2ws, point_maps, depth_maps = self.preprocess_frames(object_name, image_indices)
        extrinsic = torch.inverse(input_c2ws)

        image_indices = torch.tensor(image_indices).long().unsqueeze(-1)  # [v, 1]
        scene_indices = torch.full_like(image_indices, idx)  # [v, 1]
        indices = torch.cat([image_indices, scene_indices], dim=-1)  # [v, 2]

        return {
            "image": input_images,
            "c2w": input_c2ws,
            "extrinsic": extrinsic,
            "fxfycxcy": input_fxfycxcy,
            "intrinsic": input_intrinsics,
            "index": indices,
            "scene_name": object_name,
            "depth_map": depth_maps,
            "point_map": point_maps
        }
