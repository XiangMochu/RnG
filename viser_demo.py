import importlib
from setup import init_config
import viser 
from viser import transforms as vtf
import torch
import numpy as np 
from einops import rearrange, repeat
from utils.recover_gravity import find_best_x_rotation_zero_yaw
from model.vggt.utils.pose_enc import pose_encoding_to_extri_intri
import random
import os 
import math
from PIL import Image
import cv2
import time
import plotly.express as px


bg_img = np.linspace(250, 200, 128).astype(int)
bg_img = repeat(bg_img, 'n -> n 128 3')

white_bg = repeat(np.array([255]*3, dtype=np.uint8), 'n -> 1 1 n')

color_by_id = [
    (38, 70, 83),
    (42, 157, 143),
    (244, 162, 97),
    (193, 56, 22)
]

fixed_location = np.stack([
    np.linspace(0, 2.3*2*np.pi, 13),
    np.linspace(-np.pi*3/4, np.pi*3/4, 13)], -1)

class GSOEvalDataset:
    ''' A custom GSO dataset to prevent pose leakage'''
    def __init__(self, config):
        self.config = config 
        self.root_path = self.config.training.val_dataset_cfgs.root_dir

        with open(self.config.training.val_dataset_cfgs.split_file, 'r') as f:
            self.all_object_list = f.readlines()
        self.all_object_list = [i.strip() for i in self.all_object_list]

        random.seed(0)
        self.rand_idx = [random.sample(range(0, 25), 14) for _ in range(len(self.all_object_list))]
        self.fov = 0.6981317007977318

    def view_selector(self, idx):
        return self.rand_idx[idx]

    def __len__(self):
        return len(self.all_object_list)
    
    def __getitem__(self, idx):
        object_name = self.all_object_list[idx]
        image_indices = self.view_selector(idx)

        resize_h = self.config.model.image_tokenizer.image_size
        patch_size = self.config.model.image_tokenizer.patch_size

        img_paths = []
        images = []
        fxfycxcys = []
        for v_idx, img_idx in enumerate(image_indices):
            cur_image_path = os.path.join(self.root_path, object_name, f'{img_idx:03d}.png')
            img_paths.append(cur_image_path)
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
            images.append(image)

            focal_length = (resize_w/2) / math.tan(self.fov/2)
            fxfycxcy = torch.tensor([focal_length, focal_length, resize_w/2, resize_h/2])
            fxfycxcys.append(fxfycxcy)

        images = torch.stack(images, dim=0)
        fxfycxcys = torch.stack(fxfycxcys, dim=0)

        b, *_ = images.shape
        input_c2ws = torch.eye(4).unsqueeze(0).repeat(b, 1, 1) # dummy c2w to prevent pose leakage

        return {"image_path": img_paths, "image": images, "scene_name": object_name, 
                "fxfycxcy": fxfycxcys, "c2w": input_c2ws}


class Viewer:
    def __init__(self, config, dataset, model):
        self.config = config
        self.dataset = dataset
        self.model = model

        self.server = viser.ViserServer()
        self.server.scene.set_up_direction('-y')
        # self.server.scene.set_background_image(bg_img)
        self._init_ui()
        self.update_render_position()
        self.draw_frame()

    def _init_ui(self):
        self.current_batch_id = 0

        with self.server.gui.add_folder('Object'):
            self.slider = self.server.gui.add_slider(
                label='BatchID', min=0, max=len(self.dataset)-1, 
                step=1, initial_value=0)
            @self.slider.on_update
            def _(_):
                self.current_batch_id = int(self.slider.value)
                self.draw_frame()

            self.next_botton = self.server.gui.add_button(label='  Next ->')
            @self.next_botton.on_click
            def _(_):
                self.current_batch_id = (self.current_batch_id + 1) % len(self.dataset)
                self.slider.value = self.current_batch_id

            self.prev_botton = self.server.gui.add_button(label='<- Prev  ')
            @self.prev_botton.on_click
            def _(_):
                self.current_batch_id = (self.current_batch_id - 1) % len(self.dataset)
                self.slider.value = self.current_batch_id

            self.input_mkdown = self.server.gui.add_markdown('')

        with self.server.gui.add_folder('Camera Control'):
            self.azimuth_slider = self.server.gui.add_slider(
                label='Azimuth', min=-180, max=180, step=10, initial_value=0)
            @self.azimuth_slider.on_update
            def _(_):
                self.update_render_position()
                self.render_frame()
            
            self.elevation_slider = self.server.gui.add_slider(
                label='Elevation', min=-90, max=90, step=5, initial_value=0)
            @self.elevation_slider.on_update
            def _(_):
                self.update_render_position()
                self.render_frame()

            self.radius_slider = self.server.gui.add_slider(
                label='Radius', min=0.75, max=1.5, step=0.05, initial_value=1)
            @self.radius_slider.on_update
            def _(_):
                self.update_render_position()
                self.render_frame()
        
        with self.server.gui.add_folder('Visualization'):
            self.recon = self.server.scene.add_frame('recon', show_axes=False)
            self.recon_vis_button = self.server.gui.add_button(label='Show/Hide Reconstruction')
            @self.recon_vis_button.on_click
            def _(_):
                self.recon.visible = not self.recon.visible

            self.gen = self.server.scene.add_frame('gen', show_axes=False)
            self.gen_vis_button = self.server.gui.add_button(label='Show/Hide Generation')
            @self.gen_vis_button.on_click
            def _(_):
                self.gen.visible = not self.gen.visible

            self.already_accumulated = False
            self.accumulate_button = self.server.gui.add_button(label='Accumulate View')
            @self.accumulate_button.on_click
            def _(_):
                if self.already_accumulated:
                    print('appending to accumulated point cloud')
                    # get pc from already accumulated point cloud
                    pcd_xyz = self.accumulate_pc.points
                    pcd_rgb = self.accumulate_pc.colors
                    # get pc from current novel view
                    nv_pcd_xyz = self.current_nv_pcd.points
                    nv_pcd_rgb = self.current_nv_pcd.colors
                    nv_pcd_xyz = np.concatenate([nv_pcd_xyz, pcd_xyz], axis=0)
                    nv_pcd_rgb = np.concatenate([nv_pcd_rgb, pcd_rgb], axis=0)
                    self.accumulate_pc = self.server.scene.add_point_cloud(
                        'accum_pc', nv_pcd_xyz, nv_pcd_rgb, point_size=0.003,
                        point_shape='circle')
                else:
                    print('initializing accumulated point cloud')
                    # get pc from current novel view
                    nv_pcd_xyz = self.current_nv_pcd.points
                    nv_pcd_rgb = self.current_nv_pcd.colors

                    nv_pcd_xyz = np.concatenate([nv_pcd_xyz, *self.pts_iv], axis=0)
                    nv_pcd_rgb = np.concatenate([nv_pcd_rgb, *self.pts_color_iv], axis=0)
                    self.accumulate_pc = self.server.scene.add_point_cloud(
                        'accum_pc', nv_pcd_xyz, nv_pcd_rgb, point_size=0.003,
                        point_shape='circle')
                    self.accumulate_nv = []
                    self.already_accumulated = True
                
                print(len(self.accumulate_nv))
                accum_cam = self.server.scene.add_camera_frustum(
                    f'accum_cam/{len(self.accumulate_nv)}',
                    fov=self.current_nv_cam.fov, aspect=1,
                    scale=0.15, color=self.current_nv_cam.color,
                    image=self.current_nv_cam.image, position=self.current_nv_cam.position,
                    wxyz=self.current_nv_cam.wxyz)

                self.accumulate_nv.append(accum_cam)

            fig = px.imshow(np.ones((256, 256, 3))*256)
            self.figure = self.server.gui.add_plotly(figure=fig, aspect=1.0)
                
    @staticmethod
    def get_camera_pose(azimuth, elevation, radius=1):
        x = np.sin(azimuth) * np.cos(elevation)
        y = np.sin(elevation)
        z = np.cos(azimuth) * np.cos(elevation)
        eye = np.array([x, y, z]) * radius

        target = np.array([0,0,0])
        up = np.array([0,-1,0])
        forward = (target - eye)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        true_up = np.cross(forward, right)
        true_up /= np.linalg.norm(true_up)
        rot_matrix = np.stack([right, true_up, forward], axis=1)
        wxyz = vtf.SO3.from_matrix(rot_matrix).wxyz

        return eye, wxyz
    
    def update_render_position(self):
        azimuth = - self.azimuth_slider.value / 180 * np.pi - np.pi
        elevation = - self.elevation_slider.value / 180 * np.pi
        radius = self.radius_slider.value
        position, wxyz = self.get_camera_pose(azimuth, elevation, radius)
        if hasattr(self, 'cam_render'):
            self.cam_render.position = position
            self.cam_render.wxyz = wxyz
        else:
            self.cam_render = self.server.scene.add_camera_frustum(
                'gen/cam_render', fov=1, aspect=1, scale=0.1, line_width=3,
                position=position, wxyz=wxyz, color=(153, 78, 204))

    @staticmethod
    def erode_mask(mask, kernel=7):
        kernel = np.ones((kernel, kernel), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        return mask.astype(bool)
    
    def empty_canvas(self):
        removing_names = ['recon', 'tsdf_mesh', 'accum_pc', 'accum_cam', 'gen/pcd'] \
            + [f'source_views/cam_{i}' for i in range(4)]
        for i in removing_names:
            self.server.scene.remove_by_name(i)
        self.already_accumulated = False
        if hasattr(self, 'accumulate_pc'):
            del self.accumulate_pc
            del self.accumulate_nv

    def draw_frame(self):
        self.empty_canvas()
        batch = self.dataset[self.current_batch_id]

        img_paths = batch['image_path'][:4]
        self.input_mkdown.content = self.encode_img_path_to_mkdown(img_paths)

        with torch.no_grad(), torch.autocast(
            enabled=config.training.use_amp,
            device_type="cuda",
            dtype=amp_dtype_mapping[config.training.amp_dtype]
        ):
            batch = self.dataset[self.current_batch_id]
            batch = {k: v.cuda()[:4].unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            pose_enc = self.model.forward_pose_only(batch['image'])
        
            pred_ext, intrinsic = pose_encoding_to_extri_intri(pose_enc, (256,256))

        pred_ext = pred_ext.squeeze(0)
        ext = torch.eye(4).unsqueeze(0).repeat(pred_ext.shape[0], 1,1)

        ext[:,:3,:] = pred_ext
        c2w = torch.inverse(ext)
        c2w = c2w.numpy()

        ### recover gravity
        theta, *_ = find_best_x_rotation_zero_yaw(c2w)
        transforms_so3 = vtf.SO3.from_x_radians(theta)
        transforms_se3 = vtf.SE3.from_rotation(transforms_so3).as_matrix()
        self.gravity_rectification = vtf.SE3.from_rotation(vtf.SO3.from_x_radians(-theta)).as_matrix()
        c2w = transforms_se3[None, ...] @ c2w

        self.current_batch_c2w = c2w

        num_input_imgs = 4
        self.pts_color_iv = []
        self.pts_iv = []
        for idx in range(num_input_imgs):
            img = batch["image"][0, idx].cpu().numpy()
            img = (rearrange(img, 'c h w -> h w c') * 255).astype(np.uint8)
            wxyz_xyz = vtf.SE3.from_matrix(c2w[idx]).wxyz_xyz

            self.server.scene.add_camera_frustum(
                f'source_views/cam_{idx}', image=img, wxyz=wxyz_xyz[:4], position=wxyz_xyz[4:],
                fov=1, aspect=1, scale=0.15, color=color_by_id[idx], line_width=3)
            
            target_img, pts, pts_conf, _mask = self.forward_single_view(c2w[idx])

            mask_bg = np.sum(np.abs(img - white_bg), -1) > 0
            mask_bg = self.erode_mask(mask_bg, kernel=3)
            mask_conf = np.quantile(pts_conf[mask_bg], 0.1)
            mask = np.logical_and(pts_conf>mask_conf, mask_bg)
            pts_iv = pts[mask]
            pts_color_iv = img[mask]
            self.pts_iv.append(pts_iv)
            self.pts_color_iv.append(pts_color_iv)

            self.server.scene.add_point_cloud(f'recon/{idx}', points=pts_iv, colors=pts_color_iv,
                point_size=0.003, point_shape='circle')
        
        self.render_frame()

    def forward_single_view(self, target_c2w, use_conf_mask=False):
        target_c2w = self.gravity_rectification @ target_c2w
        target_c2w = rearrange(torch.from_numpy(target_c2w), 'i j -> 1 1 i j').float().cuda()

        with torch.no_grad(), torch.autocast(
            enabled=config.training.use_amp,
            device_type="cuda",
            dtype=amp_dtype_mapping[config.training.amp_dtype],
        ):
            batch = self.dataset[self.current_batch_id]
            batch = {k: v.cuda()[:4].unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            torch.cuda.synchronize()
            time1 = time.time()
            render_pack = self.model.forward_rendering_using_kv_cache(target_c2w)
            torch.cuda.synchronize()
            print('forward time:', time.time()-time1)
            target_img = render_pack.render
            target_img = rearrange(target_img.float(), '1 1 c h w -> h w c').detach().cpu().numpy()
            target_img = (target_img * 255).astype(np.uint8)

            pts = render_pack.points
            pts_conf = render_pack.points_conf

            pts = rearrange(pts, '1 1 c h w -> h w c').detach().cpu().numpy()
            pts_conf = rearrange(pts_conf, '1 1 1 h w -> h w').detach().cpu().numpy()

            bg_color = np.array([[[255,255,255]]])
            bg_mask = np.sum(np.abs(target_img-bg_color), -1) > 12

            if use_conf_mask:
                conf_thresh = np.quantile(pts_conf[bg_mask], 0.03)
                mask = np.logical_and((pts_conf > conf_thresh), (np.abs(pts).sum(-1) > 1e-2), bg_mask)
            else:
                mask = np.logical_and((np.abs(pts).sum(-1) > 1e-2), bg_mask)
            h,w,c = pts.shape
            pts = self.gravity_rectification[:3,:3].T @ rearrange(pts, 'h w c -> c (h w)')
            pts = rearrange(pts, 'c (h w) -> h w c', h=h, w=w)
        
        return target_img, pts, pts_conf, mask

    def render_frame(self):
        target_c2w = vtf.SE3(np.concatenate([self.cam_render.wxyz, self.cam_render.position], 0)).as_matrix()
        target_img, pts, pts_conf, mask = self.forward_single_view(target_c2w, use_conf_mask=True)
        mask = self.erode_mask(mask, kernel=13)
        pts = pts[mask]
        pts_color = target_img[mask]

        self.current_nv_pcd = self.server.scene.add_point_cloud('gen/pcd', points=pts, colors=pts_color,
            point_size=0.006, point_shape='circle')

        self.cam_render.image = target_img
        self.current_nv_cam = self.cam_render

        fig = px.imshow(target_img).update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
            ).update_layout(yaxis_title=None
            ).update_layout(xaxis_title=None
            ).update_xaxes(showticklabels=False
            ).update_yaxes(showticklabels=False)

        self.figure.figure = fig

    @staticmethod
    def pcd_to_depth(pcd, c2w):
        # pcd: h w c=3; c2w: 4 4
        h,w,c = pcd.shape
        ones = np.ones((h,w,1), dtype=pcd.dtype)
        pcd = np.concatenate([pcd, ones], axis=2)
        w2c = np.linalg.inv(c2w)
        pcd_cam = np.einsum('ij,hwj->hwi', w2c, pcd)
        depth = pcd_cam[:, :, 2] / pcd_cam[:, :, 3]
        return depth

    def encode_img_path_to_mkdown(self, image_paths):
        rows = ['Input Images:']
        num_cols = 4
        for i in range(0, len(image_paths), num_cols):
            batch = image_paths[i:i + num_cols]
            row = " | ".join(f"![image]({path})" for path in batch)
            sep = " | ".join(["---"] * len(batch))
            rows.append(row)
            rows.append(sep)
        
        return "\n".join(rows)

config = init_config()
module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
model = LVSM(config, use_kv_cache=True).cuda()
model.load_ckpt(config.training.checkpoint_dir)
model.eval()

torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32
amp_dtype_mapping = {
    "fp16": torch.float16, 
    "bf16": torch.bfloat16, 
    "fp32": torch.float32, 
    'tf32': torch.float32
}

dataset = GSOEvalDataset(config)

viewer = Viewer(config, dataset, model)
input('Press Enter to exit')
exit()