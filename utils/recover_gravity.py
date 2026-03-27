import viser 
import os 
import numpy as np
import imageio
from os.path import join as opj
from einops import rearrange, repeat
from viser import transforms as vtf
import json

# pred_folder = 'experiments/evaluation/test_obj_RnGUP_'
# pred_list = sorted(os.listdir(pred_folder))
# data_length = len(pred_list)


bg_img = np.linspace(250, 200, 128).astype(int)
bg_img = repeat(bg_img, 'n -> n 128 3')


color_by_id = [
    (38, 70, 83),
    (42, 157, 143),
    (244, 162, 97),
    (193, 56, 22)
]


import numpy as np

# ---------------- objective and minimizer ----------------
def sum_squared_yaw_for_alpha(all_poses_c2w, alpha):
    T_x = vtf.SE3.from_rotation(vtf.SO3.from_x_radians(alpha)).as_matrix()[None, ...]
    corrected = (T_x @ all_poses_c2w)
    ys = []
    for T in corrected:
        R = T[:3,:3]
        yaw = vtf.SO3.from_matrix(R).as_rpy_radians().yaw
        ys.append(yaw)
    ys = np.asarray(ys)
    ys0 = np.abs(ys)
    ys1 = np.abs(np.pi - ys)
    ys2 = np.abs(np.pi + ys)
    cost = np.minimum(ys0, ys1)
    cost = np.minimum(cost, ys2)
    return np.sum(cost), corrected, ys

def golden_section_search(f, a, b, tol=1e-8, max_iter=200):
    """
    Simple golden-section search for unimodal scalar f on [a,b].
    Returns approximate minimizer x and f(x).
    """
    gr = (np.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    fc = f(c); fd = f(d)
    it = 0
    while (b - a) > tol and it < max_iter:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) / gr
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) / gr
            fd = f(d)
        it += 1
    x = (a + b) / 2
    fx = f(x)
    return x, fx

# ---------------- public API ----------------
def find_best_x_rotation_zero_yaw(all_poses_c2w, bracket=(-np.pi/2, np.pi/2)):
    """
    Find alpha (radians) minimizing sum(yaw^2) after pre-multiplying Rx(alpha) to each pose.
    Returns:
      alpha, corrected_poses (N,4,4), yaw_array_before, yaw_array_after
    """
    poses = np.asarray(all_poses_c2w)
    if poses.ndim != 3 or poses.shape[1:] != (4,4):
        raise ValueError("all_poses_c2w must be shape (N,4,4)")

    # Precompute yaw before
    yaw_before = []
    for T in poses:
        psi = vtf.SE3.from_matrix(T).rotation().as_rpy_radians().yaw
        yaw_before.append(psi)
    yaw_before = np.array(yaw_before)

    def obj(alpha):
        s,_,_ = sum_squared_yaw_for_alpha(poses, alpha)
        return s

    a,b = bracket
    alpha_opt, _ = golden_section_search(obj, a, b, tol=1e-10, max_iter=400)

    s, corrected, yaw_after = sum_squared_yaw_for_alpha(poses, alpha_opt)
    return alpha_opt, corrected, yaw_before, yaw_after



class Viewer:
    def __init__(self):
        self.server = viser.ViserServer()
        self.server.scene.set_up_direction('-y')
        self.server.scene.set_background_image(bg_img)
        self._init_ui()
        self.draw_frame()
    
    def _init_ui(self):
        self.current_batch_id = 0
        self.slider = self.server.gui.add_slider(
            label='BatchID', min=0, max=data_length-1, 
            step=1, initial_value=0)
        @self.slider.on_update
        def _(_):
            self.current_batch_id = int(self.slider.value)
            self.draw_frame()

        self.next_botton = self.server.gui.add_button(label='  Next ->')
        @self.next_botton.on_click
        def _(_):
            self.current_batch_id = (self.current_batch_id + 1) % data_length
            self.slider.value = self.current_batch_id

        self.prev_botton = self.server.gui.add_button(label='<- Prev  ')
        @self.prev_botton.on_click
        def _(_):
            self.current_batch_id = (self.current_batch_id - 1) % data_length
            self.slider.value = self.current_batch_id

    def recover_gravity(self, pose):
        theta, *_ = find_best_x_rotation_zero_yaw(pose)
        print(theta)

        transforms_so3 = vtf.SO3.from_x_radians(theta)
        transforms_se3 = vtf.SE3.from_rotation(transforms_so3).as_matrix()[None, ...]
        recover = transforms_se3 @ pose

        for one_pose in pose:
            print(vtf.SE3.from_matrix(one_pose).rotation().as_rpy_radians())
        for one_pose in recover:
            print(vtf.SE3.from_matrix(one_pose).rotation().as_rpy_radians())
        return recover
            

    def draw_frame(self):
        obj_name = pred_list[self.current_batch_id]
        obj_pred_folder = opj(pred_folder, obj_name)
        json_path = opj(obj_pred_folder, 'metrics_pose.json')
        infer_pose = json.load(open(json_path, 'r'))
        
        input_frames = imageio.imread(opj(obj_pred_folder, 'input.png'))
        input_frames = rearrange(input_frames, 'h (n w) c -> n h w c', n=4)

        gt_pose = np.array(infer_pose['pose']['gt_pose'])
        # pred_pose = np.array(infer_pose['pose']['pred_pose'])

        pred_pose = self.recover_gravity(gt_pose)
        for i in range(4):
            gt_se3 = vtf.SE3.from_matrix(gt_pose[i])
            pred_se3 = vtf.SE3.from_matrix(pred_pose[i])
            self.server.scene.add_camera_frustum(
                f'gt/{i}', fov=1, aspect=1, image=input_frames[i], scale=0.1,
                wxyz=gt_se3.rotation().wxyz, position=gt_se3.translation(),
                color=color_by_id[i])
            
            self.server.scene.add_camera_frustum(
                f'pred/{i}', fov=1, aspect=1, scale=0.1,
                wxyz=pred_se3.rotation().wxyz, position=pred_se3.translation(),
                color=color_by_id[i])
            
if __name__ == '__main__':
    viewer = Viewer()
    input('Press Enter to exit')