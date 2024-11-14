import json
import logging
import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as F
import tqdm
from models.common import *
from argparse import ArgumentParser
from exp_runner import Runner
import trimesh
import open3d as o3d
from pathlib import Path
import os
from frnn_helper import FRNN
from models.fields import * 
from models.renderer import extract_geometry, sample_pdf
# target index does not move 
# all raw RT and fine RT should be applied to source index
# follow the logic with the robust icp
def load_cameras_and_images(images_path, masks_path, camera_params_path, frames_count, with_fixed_camera=False,
                            camera_params_list=None, pic_mode="png"):  # assmue load from a json file
    print("---------------------Loading image data-------------------------------------")
    global_K, global_M = None, None
    if with_fixed_camera and camera_params_list is not None:
        # in this case, we assume all frames share with the same K & M
        global_K = camera_params_list['K']
        global_M = camera_params_list['M']
    else:   # not pre-defined list
        with open(camera_params_path, "r") as json_file:
            camera_params_list = json.load(json_file)
    images, masks, cameras_K, cameras_M = [], [], [], []  # cameras_M should be c2w mat
    for i in range(0, frames_count + 1):
        picture_name = f"{i:03}"
        image_I_path = images_path + "/" + picture_name + "." + pic_mode
        image = cv.imread(image_I_path)
        images.append(np.array(image))
        mask_I_path = masks_path + "/" + picture_name + "." + pic_mode
        mask = cv.imread(mask_I_path)
        masks.append(np.array(mask))
        if with_fixed_camera:
            cameras_K.append(np.array(global_K))
            cameras_M.append(np.array(global_M))
        else:
            cameras_name = str(i)
            camera_K = camera_params_list[cameras_name + "_K"]
            cameras_K.append(np.array(camera_K))
            camera_M = camera_params_list[cameras_name + "_M"]
            cameras_M.append(np.array(camera_M))
    print("---------------------Load image data finished-------------------------------")
    return images, masks, cameras_K, cameras_M  # returns numpy arrays

def generate_rays_with_K_and_M(transform_matrix, intrinsic_mat, W, H, resolution_level=1):  # transform mat should be c2w mat, and numpy as input
    intrinsic_mat_inv = torch.inverse(intrinsic_mat)
    tx = torch.linspace(0, W - 1, W // resolution_level)
    ty = torch.linspace(0, H - 1, H // resolution_level)
    pixels_x, pixels_y = torch.meshgrid(tx, ty)
    p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
    p = torch.matmul(intrinsic_mat_inv[None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
    rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
    rays_v = torch.matmul(transform_matrix[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
    rays_o = transform_matrix[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3, start from transform
    return rays_o.transpose(0, 1), rays_v.transpose(0, 1)  # H W 3

def generate_all_rays(imgs, masks, cameras_K, cameras_c2w, W_all, H_all):
    # this function generate rays from given img and camera_K & c2w, also returns rays_gt as reference
    # assume input raw images are 255-uint, this function transformed to 1.0-up float0
    # stack the result into [frames_count, W*H, 3] format, assume all frames has the same resolution with W, H\
    frames_count = len(imgs)
    rays_o_all, rays_v_all, rays_gt_all, rays_mask_all = [], [], [], []
    for i in range(0, frames_count):
        rays_gt, rays_mask = imgs[i], masks[i]  ## check if is  H, W, 3
        rays_gt = rays_gt / 255.0
        rays_gt = rays_gt.reshape(-1, 3)
        rays_gt = torch.from_numpy(rays_gt.astype(np.float32)).to("cuda")
        rays_mask = rays_mask / 255.0
        rays_mask = np.where(rays_mask > 0, 1, 0).reshape(-1, 3)
        rays_mask = torch.from_numpy(rays_mask.astype(np.bool_)).to("cuda")
        rays_o, rays_v = generate_rays_with_K_and_M(cameras_c2w[i], cameras_K[i], W_all, H_all)  ## check if is  H, W, 3
        rays_o = rays_o.reshape(-1, 3)
        rays_v = rays_v.reshape(-1, 3)
        rays_o_all.append(rays_o)
        rays_v_all.append(rays_v)
        rays_gt_all.append(rays_gt)
        rays_mask_all.append(rays_mask)
    return rays_o_all, rays_v_all, rays_gt_all, rays_mask_all  # returns formulate by frames

def transfer_points_to_local_axis(points, quaternion, translation, device="cuda"):
    # convert to 4X4 mat
    w, x, y, z = quaternion
    rotate_mat = torch.zeros((3, 3), device=device)
    rotate_mat[0, 0] = 1 - 2 * (y ** 2 + z ** 2)
    rotate_mat[0, 1] = 2 * (x * y - z * w)
    rotate_mat[0, 2] = 2 * (x * z + y * w)
    rotate_mat[1, 0] = 2 * (x * y + z * w)
    rotate_mat[1, 1] = 1 - 2 * (x ** 2 + z ** 2)
    rotate_mat[1, 2] = 2 * (y * z - x * w)
    rotate_mat[2, 0] = 2 * (x * z - y * w)
    rotate_mat[2, 1] = 2 * (y * z + x * w)
    rotate_mat[2, 2] = 1 - 2 * (x ** 2 + y ** 2)
    transform_matrix = torch.zeros((4, 4), device=device)
    transform_matrix[0:3, 0:3] = rotate_mat
    transform_matrix[0:3, 3] = translation
    transform_matrix[3, 3] = 1.0
    points = torch.matmul(transform_matrix[None, :3, :3], points[:, :, None]).squeeze(dim=-1)  # N, 3
    points = points + transform_matrix[None, :3, 3]
    return points

class HonkaiStart(torch.nn.Module):
    def __init__(self, setting_json_path, case_name='Elysia', is_continue=False):
        super(HonkaiStart, self).__init__()
        self.device = 'cuda'
        with open(setting_json_path, "r") as json_file:
            reg_data = json.load(json_file)

        self.objects_cnt = reg_data['objects_cnt']
        self.train_iters = reg_data['train_iters']
        self.batch_size = reg_data['batch_size']
        self.source_index = reg_data['source_index']
        self.target_index = reg_data['target_index']
        self.sdf_batch_size = reg_data['sdf_batch_size']
        self.raw_translation = torch.tensor(reg_data['raw_translation'], dtype=torch.float32, requires_grad=True) # this para is raw from bbox align
        self.raw_quaternion  = torch.tensor(reg_data['raw_quaternion'] , dtype=torch.float32, requires_grad=True)

        self.objects, self.obj_masks, self.obj_names, self.frnns, self.zero_sdf_points_all, self.zero_sdf_points_all_mask = [], [], [], [], [], [] # all arrays have the number of objects 
        for index in range (0, self.objects_cnt):
            current_name = str(index)
            current_obj_conf_path = reg_data['obj_confs'][current_name + "_conf"]
            current_obj_name = reg_data['obj_confs'][current_name + "_name"]
            current_npz_name = reg_data['obj_confs'][current_name + "_npz"]
            current_point_mask_name = reg_data['obj_confs'][current_name + "_npz_mask"]
            current_exp_runner = Runner.get_runner(current_obj_conf_path, current_obj_name, is_continue=True)
<<<<<<< HEAD
            # current_frnn_tree = FRNN.get_frnn_tree(current_npz_name, point_mask_path=current_point_mask_name)
            # current_zero_sdf_points_mask_N3 = current_frnn_tree.mask.unsqueeze(1).repeat(1, 3) # reshape 2 N*3
            # current_zero_sdf_points, mask4zero_sdf_points, _ = current_exp_runner.split_zero_sdf_points(current_frnn_tree.unmasked_points, current_zero_sdf_points_mask_N3) # on torch
=======
            current_frnn_tree = FRNN.get_frnn_tree(current_npz_name, point_mask_path=current_point_mask_name)
            current_zero_sdf_points_mask_N3 = current_frnn_tree.mask.unsqueeze(1).repeat(1, 3) # reshape 2 N*3
            current_zero_sdf_points, mask4zero_sdf_points, _ = current_exp_runner.split_zero_sdf_points(current_frnn_tree.unmasked_points, current_zero_sdf_points_mask_N3) # on torch
>>>>>>> d367ee09b9ce182c930228aa5975b6f7de2e0079
            # pack this neus as a exp_runner in neus
            self.objects.append(current_exp_runner)
            current_sum = torch.sum(current_exp_runner.dataset.images, dim=-1)
            current_mask = (current_sum > 0.02)
            self.obj_masks.append(current_mask)
            self.obj_names.append(current_obj_name)
<<<<<<< HEAD
            # self.frnns.append(current_frnn_tree)
            # self.zero_sdf_points_all.append(current_zero_sdf_points)
            # self.zero_sdf_points_all_mask.append(mask4zero_sdf_points)
        self.W, self.H = self.obj_masks[0].shape[2], self.obj_masks[0].shape[1] # notice the index, self.obj_masks contains N sets of masks

        #align_frnn
        # transformed_source_points, mask4zero_sdf_points = None, self.zero_sdf_points_all_mask[self.source_index]
        # transformed_source_points = torch.cat(self.zero_sdf_points_all[self.source_index], dim=0).reshape([-1, 3]) # to [M, 3]

        self.target_frnn = FRNN(npz_file_path='./exp/dragon_pos1/dragon_pos1_full.ply', point_mask_path=None)
        data = o3d.io.read_point_cloud('./exp/dragon_pos2/dragon_pos2_full.ply')
        source_points = np.asarray(data.points, dtype=np.float32)
        source_points = torch.from_numpy(source_points).to('cuda')

        transform_matrix, _ = self.get_transform_matrix(translation=self.raw_translation, quaternion=self.raw_quaternion)
        # transformed_source_points = (transform_matrix[None, :3, :3] @ transformed_source_points[:, :, None]).squeeze(dim=-1)
        source_points = (transform_matrix[None, :3, :3] @ source_points[:, :, None]).squeeze(dim=-1)

        # T1_expand = (transform_matrix[0:3, 3]).repeat(len(transformed_source_points), 1)        
        T1_expand = (transform_matrix[0:3, 3]).repeat(len(source_points), 1)
        # transformed_source_points = transformed_source_points + T1_expand
        source_points = source_points + T1_expand
        # self.aligned_source_points = transformed_source_points # on cuda 
        # self.aligned_target_points = self.frnns[self.target_index].points # on cuda 
        # transformed_source_points = transformed_source_points.detach().cpu().numpy() # to numpy in np
        source_points_np = source_points.detach().cpu().numpy()
        # self.aligned_frnn_source = FRNN.get_frnn_tree_with_PDP(np_points=transformed_source_points, 
        #                                                        point_mask_path=None) # this is masked so no need for point masks
        self.source_frnn = FRNN.get_frnn_tree_with_PDP(np_points=source_points_np, point_mask_path=None)
        # write out the aligned source points
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(source_points_np)
        o3d.io.write_point_cloud('./exp/dragon_pos2/dragon_pos2_aligned.ply', point_cloud)

=======
            self.frnns.append(current_frnn_tree)
            self.zero_sdf_points_all.append(current_zero_sdf_points)
            self.zero_sdf_points_all_mask.append(mask4zero_sdf_points)
        self.W, self.H = self.obj_masks[0].shape[2], self.obj_masks[0].shape[1] # notice the index, self.obj_masks contains N sets of masks

        #align_frnn
        transformed_source_points, mask4zero_sdf_points = None, self.zero_sdf_points_all_mask[self.source_index]
        transformed_source_points = torch.cat(self.zero_sdf_points_all[self.source_index], dim=0).reshape([-1, 3]) # to [M, 3]
        transform_matrix, _ = self.get_transform_matrix(translation=self.raw_translation, quaternion=self.raw_quaternion)
        transformed_source_points = (transform_matrix[None, :3, :3] @ transformed_source_points[:, :, None]).squeeze(dim=-1)
        T1_expand = (transform_matrix[0:3, 3]).repeat(len(transformed_source_points), 1)        
        transformed_source_points = transformed_source_points + T1_expand
        self.aligned_source_points = transformed_source_points # on cuda 
        self.aligned_target_points = self.frnns[self.target_index].points # on cuda 
        transformed_source_points = transformed_source_points.detach().cpu().numpy() # to numpy in np
        self.aligned_frnn_source = FRNN.get_frnn_tree_with_PDP(np_points=transformed_source_points, 
                                                               point_mask_path=None) # this is masked so no need for point masks
>>>>>>> d367ee09b9ce182c930228aa5975b6f7de2e0079
        # import pdb; pdb.set_trace()
        
        # Dstill
        # create descendant network      
        params_to_train = []
        self.student_sdf_network = getDefaultSDF_Network() # same as neus init
        params_to_train += list(self.student_sdf_network.parameters())
        self.is_continue = is_continue
        
        self.iter_step = 0
        self.end_iter = 400000
        self.report_freq = 500
        self.validate_freq = 5000
        self.save_freq = 10000
        self.base_exp_dir = Path('exp') / 'distill' / case_name
        self.learning_rate_alpha = 0.05
        self.learning_rate = 5e-4
        self.igr_weight = 0.1
        
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]
        
        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

    def save_checkpoint(self):
        checkpoint = {
            'sdf_network_fine': self.student_sdf_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }
        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint,
                   os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def load_checkpoint(self, checkpoint_name):
            
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.student_sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.iter_step = checkpoint['iter_step']
        if checkpoint_name == "ckpt_300000.pth": # a debug neus teacher ckpt
            print_blink("load 30w sp ckpt from teacher neus")
            self.student_sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
            return
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        logging.info('End')
        
    def get_transform_matrix(self, translation, quaternion):
        w, x, y, z = quaternion
        transform_matrix = torch.tensor([
            [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w), translation[0]],
            [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w), translation[1]],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2), translation[2]],
            [0, 0, 0, 1.0]
        ], device=self.device, requires_grad=True, dtype=torch.float32)
        transform_matrix_inv = torch.inverse(transform_matrix)  # make an inverse
        transform_matrix_inv.requires_grad_(True)
        return transform_matrix, transform_matrix_inv
    
    def query_background_sdf(self, pts: torch.Tensor):
        sdf = self.runner_background.sdf_network.sdf(pts).contiguous()
        sdf_grad = self.runner_background.sdf_network.gradient(pts).squeeze().contiguous()
        return sdf, sdf_grad
    
    def generate_samples(self, source_index, image_index, samples_num = 10000, is_random=False): 
        # generate random rays from the rays all, with rays_gt and rays_o, rays_d
        # assume this is generated within mask, and is the inter-section of the target 
        if is_random:
            random_indexes = torch.randint(low=0, high=self.W, size=[samples_num]) # reflect this indexes to rays_all
            return None
        else: # generate all rays within the mask for the specified image_index
            tmp_dataset = self.objects[source_index].dataset # 
            H, W = tmp_dataset.H, tmp_dataset.W
            K, M = tmp_dataset.intrinsics_all[image_index], tmp_dataset.pose_all[image_index]
            rays_o, rays_v = generate_rays_with_K_and_M(transform_matrix=M, intrinsic_mat=K, W=W, H=H) # [H, W, 1] same as neus format
            rays_gt = tmp_dataset.images[image_index].to(self.device)
            return rays_o, rays_v, rays_gt # all should in cuda device
            # generate full rays in this pose 
            
    def calc_equivalent_camera_position(self, R, T, camera_c2w):
        with torch.no_grad():
            transform_matrix, transform_matrix_inv = self.get_transform_matrix(quaternion=R, translation=T)
            calc_equ_c2w = torch.matmul(transform_matrix, camera_c2w)
            return calc_equ_c2w
    
    def refine_rt_forward(self, source_index, target_index, vis_folder=None, write_out="full", iter_id = 0, 
                          images_total = -1, start_index=0, single_image_refine=False): # refine the rt from target_index to source_index
        # raw_quad, raw_trans are set from the init or the optimizer
        if images_total < 0:
            images_total = len(self.obj_masks[source_index]) # as default
        # print_info("running with ", images_total, "in total")
        global_loss, visited = 0, False
        neus_standard, neus_to_aligin = self.objects[source_index], self.objects[target_index]
        aabb_standard, aabb_to_aligin = self.frnns[source_index], self.frnns[target_index]
        debug_rgb = []
        for image_index in range(start_index, start_index + images_total):
            rays_o, rays_d, rays_gt = self.generate_samples(source_index=source_index, image_index=image_index)
            rays_mask = self.obj_masks[source_index][image_index]
            orgin_mat_c2w = neus_standard.dataset.pose_all[image_index]
            # reshape is used for after mask, it become [rays_sum*3]
            rays_o, rays_d, rays_gt = rays_o[rays_mask].reshape(-1, 3), rays_d[rays_mask].reshape(-1, 3), rays_gt[rays_mask].reshape(-1, 3) 
            # mask again because some edge points does not reached the zero sdf surface
            current_zero_sdf_points = self.zero_sdf_points_all[source_index][image_index] # zero sdf points from the camera rays o for the object of [source_index]
            rays_special_mask = self.zero_sdf_points_all_mask[source_index][image_index]
            rays_o, rays_d, rays_gt = rays_o[rays_special_mask].reshape(-1, 3), rays_d[rays_special_mask].reshape(-1, 3), rays_gt[rays_special_mask].reshape(-1, 3) 
            # current_zero_sdf_points = torch.from_numpy(current_zero_sdf_points.astype(np.float32)).to('cuda') trans to cuda if use aabb
            transed_zero_sdf_points = transfer_points_to_local_axis(points=current_zero_sdf_points, quaternion=self.raw_quaternion, translation=self.raw_translation)
            # query if some zero point sdf points after applying RT, transed_zero_sdf_points are now in aabb_to_aligin space
            threshold = 1e-6 # this is a square distance
            # special_mask = torch.abs_(neus_to_aligin.query_points_sdf(transed_zero_sdf_points)) < 1e-2
            # counts = np.array(aabb_to_aligin.count_points_within_threshold_batch(transed_zero_sdf_points, threshold=threshold))
            # counts, _ = aabb_to_aligin.query_nearest_points(transed_zero_sdf_points)
            counts = aabb_to_aligin.judge_points_within_ball(transed_zero_sdf_points, ball_dist=threshold, max_init_points=200, threshold_cnt=150)
            # filter_mask = counts < threshold
            filter_mask = counts 
            rays_o, rays_d, rays_gt = rays_o[filter_mask].reshape(-1, 3), rays_d[filter_mask].reshape(-1, 3), rays_gt[filter_mask].reshape(-1, 3) # remask the target points
            rays_sum = len(rays_o)
            true_indices_filter_mask = torch.nonzero(filter_mask).squeeze() # -1,3 -> -1,1
            rays_special_mask_1d = rays_special_mask[:,0]
            debug_mask = torch.zeros(rays_special_mask[:,0].size(0), dtype=torch.bool) # shape as -1, 3 in rays_mask, filled with final selected points
            num_true_b = true_indices_filter_mask.numel()
            debug_mask[true_indices_filter_mask[:num_true_b]] = True
            print_ok("decteced ", rays_sum, " rays within the square-thereshold ", threshold)  
            # import pdb; pdb.set_trace()
            # in both avaible area, cacluate using initial AABB tree, 
            if len(rays_o) > 0: # have points exists in both-aviable area
                visited = True # whether excute optimizing in refine_rt_forward
                for rays_o_batch, rays_d_batch, rays_gt_batch in zip(rays_o.split(self.batch_size),rays_d.split(self.batch_size), rays_gt.split(self.batch_size)):
                    near, far = neus_to_aligin.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)            
                    background_rgb = None
                    render_out = neus_to_aligin.renderer.render_dynamic(rays_o=rays_o_batch, rays_d=rays_d_batch,
                                                                                near=near, far=far,
                                                                                T=self.raw_translation, R=self.raw_quaternion,
                                                                                camera_c2w=orgin_mat_c2w,
                                                                                cos_anneal_ratio=neus_to_aligin.get_cos_anneal_ratio(),
                                                                                background_rgb=background_rgb)
                    color_fine = render_out["color_fine"]
                    debug_rgb.append(color_fine.clone().detach().cpu().numpy())
                    threshold = 0.001  # below threshold value will be zeros, others will be ones
                    color_fine_01 =    torch.sigmoid((color_fine - threshold) * 6480.0)
                    rays_gt_01_batch = torch.sigmoid((rays_gt_batch - threshold) * 6480.0)
                    mask_error = (color_fine_01 - rays_gt_01_batch)
                    # mask_fine_loss = F.l1_loss(color_fine_01, torch.ones_like(color_fine_01),reduction='sum') / rays_sum / 3 # normalize within cnt 
                    mask_fine_loss = F.l1_loss(mask_error, torch.zeros_like(mask_error),reduction='sum') / rays_sum / 3.0 # normalize within cnt 
                    global_loss += mask_fine_loss.clone().detach()
                    mask_fine_loss.backward()  # img_loss for refine R & T
                    torch.cuda.synchronize()
                    del render_out
                if write_out == "full" and vis_folder is not None:
                    # import pdb; pdb.set_trace();
                    debug_out_rgb = neus_to_aligin.render_novel_image_with_RTKM(q = self.raw_quaternion.detach().cpu().numpy(), t = self.raw_translation.detach().cpu().numpy(),
                            post_fix=image_index,  return_render_out=True, intrinsic_mat=neus_standard.dataset.intrinsics_all[image_index].detach().cpu().numpy(),
                            original_mat=orgin_mat_c2w.detach().cpu().numpy(), img_W=neus_to_aligin.dataset.W, img_H=neus_to_aligin.dataset.H, resolution_level=5)
                    # debug_path = vis_folder + "/" +  self.obj_names[target_index] + "_" + str(image_index) + ".png" # vis_folder is a path item
                    debug_path = vis_folder / str(iter_id) # vis_folder is a path item
                    if not debug_path.exists():
                        os.makedirs(debug_path)
                    cv.imwrite((str(debug_path) + "/" +  self.obj_names[target_index] + "_" + str(image_index) + ".png" ), debug_out_rgb)
                elif write_out == "fast" and vis_folder is not None: 
                # fast means only render in the mask, same as genshin_nerf 
                    black_there_hold = 0
                    W, H, cnt_in_mask, cnt_in_sp_mask, cnt_in_rgb = self.W, self.H, 0, 0, 0
                    debug_rgb = (np.concatenate(debug_rgb, axis=0).reshape(-1, 3) * 256).clip(0, 255).astype(np.uint8)
                    debug_img = np.zeros([H, W, 3]).astype(np.uint8)
                    for index in range(0, H):
                        for j in range(0, W):
                            if rays_mask[index][j]: # in the mask of the dataset
                                if rays_special_mask_1d[cnt_in_mask]: # index cnt_in_mask also in the special mask
                                    if filter_mask[cnt_in_sp_mask]: # finally in the filter_mask, which means contains enough neighbors
                                        if debug_rgb[cnt_in_rgb][0] > black_there_hold:
                                            debug_img[index][j][0] = debug_rgb[cnt_in_rgb][0] # store as in debug_image
                                            debug_img[index][j][1] = debug_rgb[cnt_in_rgb][1]
                                            debug_img[index][j][2] = debug_rgb[cnt_in_rgb][2]
                                        else: # write unfitted rgb as white
                                            debug_img[index][j][0] = 255
                                            debug_img[index][j][1] = 255
                                            debug_img[index][j][2] = 255
                                        cnt_in_rgb = cnt_in_rgb + 1 # move the index in debug rgb image to the nex one 
                                    else: # outside the filter_mask , highlighted as blue
                                        debug_img[index][j][0] = 255
                                        debug_img[index][j][1] = 0
                                        debug_img[index][j][2] = 0
                                    cnt_in_sp_mask = cnt_in_sp_mask + 1
                                else: # outside the  special mask, highlighted as green
                                    debug_img[index][j][0] = 0
                                    debug_img[index][j][1] = 255
                                    debug_img[index][j][2] = 0   
                                cnt_in_mask = cnt_in_mask + 1
                    print_blink("saving debug image at " + str(iter_id) + "th validation, with image inedex " + str(image_index))
                    cv.imwrite((vis_folder / (str(iter_id) + "_" + str(image_index) + ".png")).as_posix(), debug_img)
                    # write correspond ply to check result
                    # point_cloud = o3d.geometry.PointCloud() # auto write out
                    # import pdb; pdb.set_trace()
                    # store_path = (vis_folder / (str(iter_id) + "_" + str(image_index) + "cur.ply"))
                    # point_cloud.points = o3d.utility.Vector3dVector(current_zero_sdf_points.clone().detach().cpu().numpy())
                    # o3d.io.write_point_cloud(str(store_path), point_cloud)
                    # store_path = (vis_folder / (str(iter_id) + "_" + str(image_index) + "tra.ply"))
                    # point_cloud.points = o3d.utility.Vector3dVector(transed_zero_sdf_points.clone().detach().cpu().numpy())
                    # o3d.io.write_point_cloud(str(store_path), point_cloud)
                else: 
                    print("no debug output")
            else:
                continue
            # count = 0
            if single_image_refine:
                print_info("acced loss at image index ", image_index, " with 01 loss", global_loss)
            else:
                print_info("calc loss at this index ", image_index, " with 01 loss", global_loss, "and return immediately")
                R_ek_loss = torch.abs(torch.norm(self.raw_quaternion) - 1)
                # print_blink("R_ek_loss ", str(R_ek_loss.clone().detach().cpu().numpy()))
                R_ek_loss.backward()
                global_loss = R_ek_loss + global_loss
                return global_loss # return to optimize single
        R_ek_loss = torch.abs(torch.norm(self.raw_quaternion) - 1)
        R_ek_loss.backward()
        global_loss = R_ek_loss + global_loss
        return global_loss, visited
    
    def render_colored_single_ply(self, neus_index, vis_folder=None, images_total = -1): 
        # render a colored ply for the final reg-result
        if images_total < 0:
            images_total = len(self.obj_masks[neus_index])  # as default
        neus_standard = self.objects[neus_index]
        points_all, colors_all = [], []
        for image_index in range(0, images_total):
            debug_rgb = []
            rays_o, rays_d, rays_gt = self.generate_samples(source_index=neus_index, image_index=image_index)
            rays_mask = self.obj_masks[neus_index][image_index]
            # reshape is used for after mask, it become [rays_sum*3]
            rays_o, rays_d, rays_gt = rays_o[rays_mask].reshape(-1, 3), rays_d[rays_mask].reshape(-1, 3), rays_gt[
                rays_mask].reshape(-1, 3)
            # mask again because some edge points does not reached the zero sdf surface, notice that current_zero_sdf_points has been masked
            current_zero_sdf_points = self.zero_sdf_points_all[neus_index][image_index] # zero sdf points from the camera rays o for the object of [source_index]， 
            rays_special_mask = self.zero_sdf_points_all_mask[neus_index][image_index]
            rays_o, rays_d, rays_gt = rays_o[rays_special_mask].reshape(-1, 3), rays_d[rays_special_mask].reshape(-1,3), rays_gt[rays_special_mask].reshape(-1, 3)
            rays_special_mask_1d = rays_special_mask[:, 0]
            if len(rays_o) > 0:  # have points exists in both-aviable area
                for rays_o_batch, rays_d_batch in zip(rays_o.split(self.batch_size),rays_d.split(self.batch_size)):
                    near, far = neus_standard.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
                    #just render to get color, do not use dynamic thing 
                    render_out = neus_standard.renderer.render(rays_o=rays_o_batch, rays_d=rays_d_batch,near=near, far=far,
                                                                cos_anneal_ratio=neus_standard.get_cos_anneal_ratio(),background_rgb=None)
                    color_fine = render_out["color_fine"]
                    debug_rgb.append(color_fine.clone().detach().cpu().numpy())
                    torch.cuda.synchronize()
                    del render_out
                black_there_hold = 0
                W, H, cnt_in_mask, cnt_in_rgb = self.W, self.H, 0, 0
                debug_rgb = (np.concatenate(debug_rgb, axis=0).reshape(-1, 3) * 256).clip(0, 255).astype(np.uint8)
                debug_img = np.zeros([H, W, 3]).astype(np.uint8)
                # feed rendered result
                for index in range(0, H):
                    for j in range(0, W):
                        if rays_mask[index][j]:  # in the mask of the dataset
                            if rays_special_mask_1d[cnt_in_mask]:  # index cnt_in_mask also in the special mask
                                if debug_rgb[cnt_in_rgb][0] > black_there_hold:
                                    debug_img[index][j][0] = debug_rgb[cnt_in_rgb][0]  # store as in debug_image
                                    debug_img[index][j][1] = debug_rgb[cnt_in_rgb][1]
                                    debug_img[index][j][2] = debug_rgb[cnt_in_rgb][2]
                                    debug_rgb2bgr = np.array([debug_rgb[cnt_in_rgb][2], debug_rgb[cnt_in_rgb][1], debug_rgb[cnt_in_rgb][0]])
                                    # ensure only add colored points
                                    colors_all.append(debug_rgb2bgr) # this is numpy, using debug_rgb2bgr
                                    points_all.append(current_zero_sdf_points[cnt_in_rgb].clone().detach().cpu().numpy()) # correspond points in this 
                                else:  # write unfitted rgb as white
                                    debug_img[index][j][0] = 255
                                    debug_img[index][j][1] = 255
                                    debug_img[index][j][2] = 255
                                cnt_in_rgb = cnt_in_rgb + 1
                            else:  # outside the  special mask, highlighted as green
                                debug_img[index][j][0] = 0
                                debug_img[index][j][1] = 255
                                debug_img[index][j][2] = 0
                            cnt_in_mask = cnt_in_mask + 1
                # print_blink("saving debug image with image inedex " + str(image_index))
                # cv.imwrite((vis_folder / (str(image_index) + ".png")).as_posix(), debug_img)            
            else:
                continue
            print_info("calc points with rgb at this index ", image_index)
        if vis_folder is not None: # write out colored ply for debug
            point_cloud = o3d.geometry.PointCloud() # auto write out
            point_cloud.points = o3d.utility.Vector3dVector(np.asarray(points_all, dtype=np.float32))
            point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors_all, dtype=np.float32) / 255.0)
            if not os.path.exists(vis_folder):
                os.makedirs(vis_folder)
            vis_path = str(vis_folder) + "/" + str(neus_index) + ".ply"
            o3d.io.write_point_cloud(vis_path, point_cloud, write_ascii=True)
        return points_all, colors_all

    # genearte query points from teacher model and returns sdf points
    def generate_sample_sdfs_from_teacher_model(self, neus_index : int, sample_nums=512, genarate_option="mix", image_index=-1):
        samples, sdfs, samples3d = None, None, None
        teacher_neus = self.objects[neus_index] # this is a packed runner
        if genarate_option == "mix": # mix sample, assmuing image index is legeal
            rays_o, rays_d, rays_gt = self.generate_samples(source_index=neus_index, image_index=image_index)
            rays_mask = self.obj_masks[neus_index][image_index] # reshape is used for after mask, it become [rays_sum*3]
            rays_o, rays_d, rays_gt = rays_o[rays_mask].reshape(-1, 3), rays_d[rays_mask].reshape(-1, 3), rays_gt[rays_mask].reshape(-1, 3)
            # mask again because some edge points does not reached the zero sdf surface, notice that current_zero_sdf_points has been masked
            current_zero_sdf_points = self.zero_sdf_points_all[neus_index][image_index] # zero sdf points from the camera rays o for the object of [source_index]， 
            rays_special_mask = self.zero_sdf_points_all_mask[neus_index][image_index]
            rays_o, rays_d, rays_gt = rays_o[rays_special_mask].reshape(-1, 3), rays_d[rays_special_mask].reshape(-1,3), rays_gt[rays_special_mask].reshape(-1, 3)
            light_walk_distance = torch.norm(current_zero_sdf_points - rays_o, dim=1, keepdim=True) # what shape

            samples_total, samples_per_ray = light_walk_distance.size(0), 10
            means = light_walk_distance.repeat(samples_per_ray, 1)
            std_devs = (light_walk_distance / 100).repeat(samples_per_ray, 1) # 0.01 stdv
            samples, samples3d = torch.normal(means, std_devs), torch.empty(samples_total * samples_per_ray, 3)
            samples = torch.clamp(samples, min=0)
            light_walk_distance = light_walk_distance.repeat_interleave(samples_per_ray, dim=0)
            samples = torch.clamp(samples, max=light_walk_distance)
            rays_o = rays_o.repeat_interleave(samples_per_ray, dim=0)
            rays_d = rays_d.repeat_interleave(samples_per_ray, dim=0)
            samples3d = rays_o + rays_d * samples
        elif genarate_option == 'sphere':
            # generate points in a sphere
            samples3d = torch.randn(2**16, 3)
            samples3d = samples3d / torch.norm(samples3d, dim=1, keepdim=True) * 0.5  # Normalize to sphere surface
            # Scale to within a sphere with radius 0.1
            samples3d = samples3d * torch.rand(2**16, 1) * 0.35
        elif genarate_option == 'neus':
            sample_sphere_raidus = 0.55
            data = self.objects[neus_index].dataset.gen_random_rays_at(image_index, self.batch_size)
            rays_o, rays_d, _, _ = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
            near, far = self.objects[neus_index].dataset.near_far_from_sphere(rays_o, rays_d, radius=sample_sphere_raidus)
            sample_dist = 2.0 / 64  # Assuming the region of interest is a unit sphere
            z_vals = torch.linspace(0.0, 1.0, 64)
            z_vals = near + (far - near) * z_vals[None, :]
            t_rand = (torch.rand([self.batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / 64
            pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
            sdf = self.student_sdf_network.sdf(pts.reshape(-1, 3)).reshape(self.batch_size, 64)
            for i in range(4):
                new_z_vals = self.up_sample(rays_o,
                                            rays_d,
                                            z_vals,
                                            sdf,
                                            64 // 4,
                                            64 * 2 ** i)
                z_vals, sdf = self.cat_z_vals(rays_o,
                                            rays_d,
                                            z_vals,
                                            new_z_vals,
                                            sdf,
                                            last=(i + 1 == 4))
            # Section length
            dists = z_vals[..., 1:] - z_vals[..., :-1]
            dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
            mid_z_vals = z_vals + dists * 0.5
            # Section midpoints
            samples3d = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
            samples3d = samples3d.reshape(-1, 3)
<<<<<<< HEAD

        # dist, _, _ = self.target_frnn.query_Knear_points(samples3d, 10)
        # average_target_dist = torch.mean(dist, dim=-1)
        # dist, _, _ = self.source_frnn.query_Knear_points(samples3d, 10)
        # average_source_dist = torch.mean(dist, dim=-1)    

        # source_samples = samples3d[average_source_dist < average_target_dist]
        # target_samples = samples3d[average_source_dist >= average_target_dist]

        # Compute SDF values for source and target
        # SDF of target network
        sdf_target = self.objects[self.target_index].sdf_network.sdf(samples3d).contiguous()
        # SDF of source network
        _, transform_inv = self.get_transform_matrix(translation=self.raw_translation, quaternion=self.raw_quaternion)
        sample3d_to_source = (transform_inv[None, :3, :3] @ samples3d[:, :, None]).squeeze(dim=-1)
        T1_expand = (transform_inv[0:3, 3]).repeat(len(sample3d_to_source), 1)
        sample3d_to_source = sample3d_to_source + T1_expand
        sdf_source = self.objects[self.source_index].sdf_network.sdf(sample3d_to_source).contiguous()

        # Create boolean masks using bitwise operations (&, |)
        mask1 = ((sdf_source > 0) & (sdf_target > 0) & (sdf_source >= sdf_target)).squeeze()
        mask2 = ((sdf_source > 0) & (sdf_target <= 0)).squeeze()
        mask3 = ((sdf_source < 0) & (sdf_target < 0) & (sdf_source >= sdf_target)).squeeze()
        source_mask = mask1 | mask2 | mask3
        # Apply masks to filter samples for `source_samples`
        source_samples = samples3d[source_mask]

        # Define masks for `target_samples`
        mask1 = ((sdf_source > 0) & (sdf_target > 0) & (sdf_target > sdf_source)).squeeze()
        mask2 = ((sdf_source <= 0) & (sdf_target > 0)).squeeze()
        mask3 = ((sdf_source < 0) & (sdf_target < 0) & (sdf_target > sdf_source)).squeeze()
        target_mask = mask1 | mask2 | mask3
        # Apply masks to filter samples for `target_samples`
        target_samples = samples3d[target_mask]
=======
        elif genarate_option == "zero": # generate zero sdfs from the incoming points:
            if neus_index == self.source_index: # need to transform the sample points to target axis
                len_mask = len(self.aligned_source_points)
                samples_from = self.aligned_source_points
            else:
                len_mask = len(self.aligned_target_points)
                samples_from = self.aligned_target_points
            sample_mask = torch.zeros(len_mask, dtype=torch.int)
            indics = torch.randperm(len())[:sample_nums] # random pick up
            sample_mask[indics] = 1
            samples3d = samples_from[sample_mask].reashape(-1, 3)
            sdfs = torch.zeros(len(sample_nums))

            # using open3d output the sample3d_to_source
            point_cloud, store_path = o3d.geometry.PointCloud(), "./exp/distill/test_samples_" + str(neus_index) + ".ply"
            point_cloud.points = o3d.utility.Vector3dVector(samples3d.clone().detach().cpu().numpy())
            o3d.io.write_point_cloud(store_path, point_cloud)

            return samples3d, sdfs

        dist, _, _ = self.frnns[self.target_index].query_Knear_points(samples3d, 30)
        average_target_dist = torch.mean(dist, dim=-1)
        dist, _, _ = self.aligned_frnn_source.query_Knear_points(samples3d, 30)
        average_source_dist = torch.mean(dist, dim=-1)        
        source_samples = samples3d[average_source_dist <  average_target_dist]
        target_samples = samples3d[average_source_dist >= average_target_dist] # target 
>>>>>>> d367ee09b9ce182c930228aa5975b6f7de2e0079
        
        if neus_index == self.target_index or genarate_option == 'sphere':
            _, transform_inv = self.get_transform_matrix(translation=self.raw_translation, quaternion=self.raw_quaternion)
            sample3d_to_source = (transform_inv[None, :3, :3] @ source_samples[:, :, None]).squeeze(dim=-1)
            T1_expand = (transform_inv[0:3, 3]).repeat(len(sample3d_to_source), 1)
            sample3d_to_source = sample3d_to_source + T1_expand
<<<<<<< HEAD
            source_sdfs = self.objects[self.source_index].sdf_network.sdf(sample3d_to_source).contiguous()
        else:
            source_sdfs = self.objects[self.source_index].sdf_network.sdf(source_samples).contiguous()
        
=======
            # using open3d output the sample3d_to_source
            # point_cloud, store_path = o3d.geometry.PointCloud(), "./exp/distill/test_samples_apply_inv" + ".ply"
            # point_cloud.points = o3d.utility.Vector3dVector(sample3d_to_source.clone().detach().cpu().numpy())
            # o3d.io.write_point_cloud(store_path, point_cloud)
            # point_cloud, store_path = o3d.geometry.PointCloud(), "./exp/distill/test_samples_" + ".ply"
            # point_cloud.points = o3d.utility.Vector3dVector(source_samples.clone().detach().cpu().numpy())
            # o3d.io.write_point_cloud(store_path, point_cloud)
        source_sdfs = self.objects[self.source_index].sdf_network.sdf(sample3d_to_source).contiguous()
>>>>>>> d367ee09b9ce182c930228aa5975b6f7de2e0079
        target_sdfs = self.objects[self.target_index].sdf_network.sdf(target_samples).contiguous()
        samples3d = torch.cat([source_samples, target_samples], dim=0)
        sdfs      = torch.cat([source_sdfs, target_sdfs], dim=0)
        return samples3d, sdfs

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere
        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples
    
    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)
        if not last:
            new_sdf = self.student_sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)
        return z_vals, sdf

    def query_student_sdfs(self, pts: torch.Tensor):
        sdf = self.student_sdf_network.sdf(pts).contiguous()
        sdf_grad = self.student_sdf_network.gradient(pts).squeeze().contiguous()
        return sdf, sdf_grad
    
    # the trained 
    def distill_sdf_forward(self, sample_points, teacher_sdfs): # calc sdf loss and EK loss for the network where needs with no grad? 
        sdf_nn_output = self.student_sdf_network(sample_points)
        sdf = sdf_nn_output[:, :1]
        student_sdfs_source_error = sdf - teacher_sdfs
        sdf_loss = F.l1_loss(student_sdfs_source_error, torch.zeros_like(student_sdfs_source_error),reduction='sum')  # calc sdf loss in neus_source
        # calc ek_loss borrow from neus trainner & render
        points_norm = torch.linalg.norm(sample_points, ord=2, dim=-1, keepdim=True)
        gradients = self.student_sdf_network.gradient(sample_points).squeeze().contiguous()
        relax_inside_sphere = (points_norm < 1.2).float().detach()
        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients, ord=2,dim=-1) - 1.0) ** 2
        Ek_loss = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)  
        # import pdb; pdb.set_trace()  
        return {
            'sdf_loss': sdf_loss,
            "Ek_loss": Ek_loss
        }
    
    def update_learning_rate(self):
        if self.iter_step < 5000:
            learning_factor = self.iter_step / 5000
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - 5000) / (self.end_iter - 5000)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def find_latest_ckpt_name(self):
        latest_model_name = None
        model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
        model_list = []
        for model_name in model_list_raw:
            if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                model_list.append(model_name)
        model_list.sort()
        latest_model_name = model_list[-1]

        return latest_model_name

    # TODO: pack a training function from honkai neus
    def train_student_sdf(self, is_continue=False):
        if is_continue:
            # load ckpt in training:
            ckpt_name = self.find_latest_ckpt_name()
            if ckpt_name is not None:
                self.load_checkpoint(ckpt_name)
            else:
                print_error("no ckpt is found")       
        self.update_learning_rate()

        res_step = self.end_iter - self.iter_step
        source_images_total, target_images_total = len(self.obj_masks[self.source_index]), len(self.obj_masks[self.target_index])
        process_bar = tqdm.tqdm(total=res_step)

        iter_i = 0
        while self.iter_step < self.end_iter:
            with torch.no_grad():
                if iter_i % 2 == 0: # swtich the teacher neus for each step
                    # Using source NeuS as teacher
                    image_perm = torch.randperm(source_images_total)
                    image_index = image_perm[(self.iter_step // 2) % len(image_perm)]
                    samples, sdfs = self.generate_sample_sdfs_from_teacher_model(neus_index=self.source_index, image_index=image_index, genarate_option='sphere')
                    # transform_matrix, _ = self.get_transform_matrix(translation=self.raw_translation, quaternion=self.raw_quaternion)
                    # # samples_source__ = torch.matmul(transform_matrix[None, :3, :3], samples_source[:, :, None]).squeeze(dim=-1)
                    # samples = (transform_matrix[None, :3, :3] @ samples[:, :, None]).squeeze(dim=-1)
                    # T1_expand = (transform_matrix[0:3, 3]).repeat(len(samples), 1)
                    # samples = samples + T1_expand # notice its important to apply transformation to samples so that we can align those points to target axis
                else:
                    # Using target Neus as teacher
                    image_perm = torch.randperm(target_images_total)
                    image_index = image_perm[(self.iter_step // 2) % len(image_perm)]
                    samples, sdfs = self.generate_sample_sdfs_from_teacher_model(neus_index=self.target_index, image_index=image_index, genarate_option='sphere')
            # print_info("sampling with points ", len(samples))
            # batching this process
            for samples_batch, sdfs_batch in zip(samples.split(self.sdf_batch_size), sdfs.split(self.sdf_batch_size)):
                res_out = self.distill_sdf_forward(sample_points=samples_batch, teacher_sdfs=sdfs_batch)
                # import pdb; pdb.set_trace()
                ek_loss = res_out['Ek_loss'] / len(samples_batch)
                sdf_loss = res_out['sdf_loss'] / len(samples_batch)  
<<<<<<< HEAD
                sdf_weight = 100
                if self.iter_step < 20000:
                    sdf_weight = 100
=======
                sdf_weight = 1000
                if self.iter_step < 20000:
                    sdf_weight = 1000
>>>>>>> d367ee09b9ce182c930228aa5975b6f7de2e0079
                loss = ek_loss * self.igr_weight + sdf_loss * sdf_weight # bigger 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.iter_step += 1
                process_bar.update(1)
                if self.iter_step % self.report_freq == 0:
                    print('iter:{:8>d} weighted loss = {} sdf_loss = {} ek_Loss = {} lr={}'.format(self.iter_step, loss, sdf_loss, ek_loss, self.optimizer.param_groups[0]['lr']))
                    #also save sample points as debug
                    # point_cloud, store_path = o3d.geometry.PointCloud(), "./debug/test_samples" + str(iter_i) + ".ply"
                    # point_cloud.points = o3d.utility.Vector3dVector(samples.clone().detach().cpu().numpy())
                    # o3d.io.write_point_cloud(store_path, point_cloud)
                if self.iter_step % self.save_freq == 0:
                    self.save_checkpoint()
                    #also save sample points as debug
                    point_cloud, store_path = o3d.geometry.PointCloud(), "./exp/distill/test_samples" + str(iter_i) + ".ply"
                    point_cloud.points = o3d.utility.Vector3dVector(samples.clone().detach().cpu().numpy())
                    o3d.io.write_point_cloud(store_path, point_cloud)
                if self.iter_step % self.validate_freq == 0:
                    self.validate_student_mesh()
                self.update_learning_rate()
            iter_i += 1
        process_bar.close()
    
    def validate_student_mesh(self, world_space=False, resolution=128, threshold=0.0, is_continue=False, bound_min=None, bound_max=None):
        if bound_min is None: 
            bound_min = torch.tensor([-0.1,-0.1,-0.1], dtype=torch.float32)
            bound_max = torch.tensor([0.1,  0.1, 0.1], dtype=torch.float32)
        if is_continue:
            # load ckpt in training:
            ckpt_name = self.find_latest_ckpt_name()
            if ckpt_name is not None:
                self.load_checkpoint(ckpt_name)
            else:
                print_error("no ckpt is found")  
        vertices, triangles = \
            extract_geometry(bound_min,
                                  bound_max, 
                                  resolution=resolution, 
                                  threshold=threshold,
                                  query_func=lambda pts: -self.student_sdf_network.sdf(pts))
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)), encoding='ascii')
        print("save at " + os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

def get_optimizer(mode, honkaiStart):
    optimizer = None
    if  mode == "refine_rt":
        optimizer = torch.optim.Adam(
            [
                {'params': getattr(honkaiStart, 'raw_translation'), 'lr': 3e-5},
                {'params': getattr(honkaiStart, 'raw_quaternion'), 'lr': 3e-5},
            ],
            amsgrad=False
        )
    elif mode == "distill":
        optimizer = torch.optim.Adam(
            [
                {'params': getattr(honkaiStart, 'raw_translation'), 'lr': 3e-5},
                {'params': getattr(honkaiStart, 'raw_quaternion'), 'lr': 3e-5},
            ],
            amsgrad=False
        )
    return optimizer

def refine_rt(honkaiStart : HonkaiStart, vis_folder=None, single_image_refine=False, single_sub_length=1, write_out=None): # runs as a train function 
    def refine_rt_forward(optimizer, iter_id=-1, start_index=-1):
        optimizer.zero_grad()
        if vis_folder != None:
            if not os.path.exists(vis_folder):
                os.makedirs(vis_folder)
        if single_image_refine: # calc single image before refine
            loss, visited = honkaiStart.refine_rt_forward(source_index = honkaiStart.source_index, target_index=honkaiStart.target_index, 
            vis_folder=vis_folder, write_out=write_out, iter_id = iter_id, single_image_refine=single_image_refine, images_total=1, start_index=start_index)
        else: # calc all images before refine
            loss, visited = honkaiStart.refine_rt_forward(source_index = honkaiStart.source_index, target_index=honkaiStart.target_index, 
                vis_folder=vis_folder, write_out=write_out, iter_id = iter_id)
        return loss, visited   
    optimizer = get_optimizer('refine_rt', honkaiStart=honkaiStart)
    train_iters = honkaiStart.train_iters
    pbar = tqdm.trange(0, train_iters)
    img_len = len(honkaiStart.objects[honkaiStart.source_index].dataset.images)
    for i in pbar:
        if single_image_refine: 
            for _ in range (0, single_sub_length):
                test_devision = 0
                loss, visited = refine_rt_forward(optimizer=optimizer, iter_id=i, start_index=(i + test_devision) % img_len)
                optimizer.step()   
                print('raw_translation: {}, raw_quaternion: {}, loss: {}'.format(honkaiStart.raw_translation, honkaiStart.raw_quaternion, loss.norm()))
                print('trans gradient : {}, quad gradient : {}, loss: {}'.format(honkaiStart.raw_translation.grad, honkaiStart.raw_quaternion.grad, loss.norm()))   
                
        else:
            loss, visited = refine_rt_forward(optimizer=optimizer, iter_id=i)
            optimizer.step()    
            print('raw_translation: {}, raw_quaternion: {}, loss: {}'.format(honkaiStart.raw_translation, honkaiStart.raw_quaternion, loss.norm()))   
        if loss.norm() < 1e-6 and visited:
            print_info("REG finished with loss ", loss.norm)
            break
        # print("refining RAW_rt from pose 0 to pose 1")
        # print('raw_translation: {}, raw_quaternion: {}, loss: {}'.format(honkaiStart.raw_translation, honkaiStart.raw_quaternion, loss.norm()))   
    return


if __name__ == '__main__':
    print_blink('Honkai NerF, start!!!')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_default_dtype(torch.float32)
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    parser = ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/json/base.json')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--vis_folder', type=str, default="debug/dragon2_FRNN")
    parser.add_argument('--write_out', type=str, default=0)
    parser.add_argument('--mode', type=str, default='refine_rt')
    parser.add_argument('--is_continue', default=False, action="store_true")
    
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    honkaiStart = HonkaiStart(args.conf)
    if args.mode == 'refine_rt':
        refine_rt(honkaiStart=honkaiStart, vis_folder= Path(args.vis_folder), single_image_refine=True, write_out=args.write_out)
    elif args.mode == "render_ply":
        honkaiStart.render_colored_single_ply(neus_index=1, vis_folder=Path(args.vis_folder))
    elif args.mode == "distill":
        honkaiStart.train_student_sdf(is_continue = args.is_continue)
    elif args.mode == "validate":
        honkaiStart.validate_student_mesh(is_continue = args.is_continue, resolution=512)
    seed = 20031012
    

"""
python reg_FRNN.py --conf ./confs/json/fuxuan.json --write_out fast --gpu 2 
python reg_FRNN.py --conf ./confs/json/klee.json --write_out fast --gpu 3
python reg_FRNN.py --conf ./confs/json/fuxuan_fricp.json --write_out fast --gpu 0
python reg_FRNN.py --conf ./confs/json/fuxuan_fricp.json --mode render_ply --write_out fast --gpu 1
python reg_FRNN.py --conf ./confs/json/fuxuan_fricp.json --mode distill --gpu 1  --is_continue
python reg_FRNN.py --conf ./confs/json/fuxuan_fricp.json --mode validate --gpu 1 --is_continue
"""
    