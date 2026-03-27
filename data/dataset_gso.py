import random
import torch
from data.dataset_objaverse import ObjaverseDataset


class GSODataset(ObjaverseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.config = config 
        self.root_path = self.config.training.val_dataset_cfgs.root_dir

        del self.all_object_list
        with open(self.config.training.val_dataset_cfgs.split_file, 'r') as f:
            self.all_object_list = f.readlines()
        self.all_object_list = [i.strip() for i in self.all_object_list]

        self.fov = 0.6981317007977318

        random.seed(0)
        self.rand_idx = [random.sample(range(0, 25), 14) for _ in range(len(self.all_object_list))]

    def view_selector(self, idx):
        return self.rand_idx[idx]

    def find_extra_indices(self, indices, num_extra):
        full_indices = set(list(range(25)))
        remain = list(full_indices - set(indices))
        extra = random.sample(remain, num_extra)
        return indices[:4] + extra + indices[4:]

    def __getitem__(self, idx):
        object_name = self.all_object_list[idx]
        image_indices = self.view_selector(idx)

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
