import json
import logging
import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as F
from icecream import ic
from tqdm import trange
from models.common import *
from argparse import ArgumentParser
from exp_runner import Runner
import time
import math
import trimesh
from pathlib import Path
import os
import torch.nn as nn
from AABB_helper import PointCloud2AABBTree

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
    # transform_matrix = torch.from_numpy(transform_matrix.astype(np.float32)).to('cuda')  # add to cuda
    intrinsic_mat_inv = torch.inverse(intrinsic_mat)
    # intrinsic_mat_inv = torch.from_numpy(intrinsic_mat_inv.astype(np.float32)).to('cuda')
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
    # returns rays_o_all, rays_v_all, rays_gt_all, rays_mask_all formulate by frames
    return rays_o_all, rays_v_all, rays_gt_all, rays_mask_all

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
    def __init__(self, setting_json_path):
        super(HonkaiStart, self).__init__()
        self.device = 'cuda'
        with open(setting_json_path, "r") as json_file:
            reg_data = json.load(json_file)
        self.objects_cnt = reg_data['objects_cnt']
        self.train_iters = reg_data['train_iters']
        self.batch_size = reg_data['batch_size']
        self.standard_index = reg_data['standard_index']
        self.to_aligin_index = reg_data['to_aligin_index']
        self.raw_translation = torch.tensor(reg_data['raw_translation'], dtype=torch.float32, requires_grad=True) # this para is raw from bbox align
        self.raw_quaternion  = torch.tensor(reg_data['raw_quaternion'] , dtype=torch.float32, requires_grad=True)
        self.objects, self.obj_masks, self.obj_names, self.aabb_trees, self.zero_sdf_points_all = [], [], [], [], [] # all arrays have the number of objects 
        for index in range (0, self.objects_cnt):
            current_name = str(index)
            current_obj_conf_path = reg_data['obj_confs'][current_name + "_conf"]
            current_obj_name = reg_data['obj_confs'][current_name + "_name"]
            current_npz_name = reg_data['obj_confs'][current_name + "_npz"]
            current_exp_runner = Runner.get_runner(current_obj_conf_path, current_obj_name, is_continue=True)
            current_aabb_tree = PointCloud2AABBTree.get_aabb_tree(current_npz_name)
            current_zero_sdf_points = current_exp_runner.split_zero_sdf_points(current_aabb_tree.points)
            # pack this neus as a exp_runner in neus
            self.objects.append(current_exp_runner)
            current_sum = torch.sum(current_exp_runner.dataset.images, dim=-1)
            current_mask = (current_sum > 0.02)
            self.obj_masks.append(current_mask)
            self.obj_names.append(current_obj_name)
            self.aabb_trees.append(current_aabb_tree)
            self.zero_sdf_points_all.append(current_zero_sdf_points)
        self.W, self.H = self.obj_masks[0].shape[2], self.obj_masks[0].shape[1] # notice the index, self.obj_masks contains N sets of masks

        
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
        # TODO: generate random rays from rays all within the mask 
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
    
    def refine_rt_forward(self, standard_index, to_aligin_index, vis_folder=None, write_out="full", iter_id = 0, 
                          images_total = -1, start_index=0, single_image_refine=False): # refine the rt from to_aligin_index to standard_index
        # raw_quad, raw_trans are set from the init or the optimizer
        # TODO:: select poses to make sure it is generate from a both-available area
        if images_total < 0:
            images_total = len(self.obj_masks[standard_index]) # as default
        # print_info("running with ", images_total, "in total")
        global_loss, visited = 0, False
        neus_standard, neus_to_aligin = self.objects[standard_index], self.objects[to_aligin_index]
        aabb_standard, aabb_to_aligin = self.aabb_trees[standard_index], self.aabb_trees[to_aligin_index]
        debug_rgb = []
        for image_index in range(start_index, start_index + images_total):
            rays_o, rays_d, rays_gt = self.generate_samples(source_index=standard_index, image_index=image_index)
            rays_mask = self.obj_masks[standard_index][image_index]
            orgin_mat_c2w = neus_standard.dataset.pose_all[image_index]
            # reshape is used for after mask, it become [rays_sum*3]
            rays_o, rays_d, rays_gt = rays_o[rays_mask].reshape(-1, 3), rays_d[rays_mask].reshape(-1, 3), rays_gt[rays_mask].reshape(-1, 3) 
            current_zero_sdf_points = self.zero_sdf_points_all[standard_index][image_index] # zero sdf points from the camera rays o for the object of [standard_index]
            # apply raw_RT to current_zero_sdf_points, send to cuda first
            current_zero_sdf_points = torch.from_numpy(current_zero_sdf_points.astype(np.float32)).to('cuda')
            transed_zero_sdf_points = transfer_points_to_local_axis(points=current_zero_sdf_points, quaternion=self.raw_quaternion, translation=self.raw_translation)
            # query if some zero point sdf points after applying RT, transed_zero_sdf_points are now in aabb_to_aligin space
            threshold = 5e-3
            # special_mask = torch.abs_(neus_to_aligin.query_points_sdf(transed_zero_sdf_points)) < 1e-2
            counts = np.array(aabb_to_aligin.count_points_within_threshold_batch(transed_zero_sdf_points, threshold=threshold))
            special_mask = counts > 5
            rays_o, rays_d, rays_gt = rays_o[special_mask], rays_d[special_mask], rays_gt[special_mask] # remask the target points
            rays_sum = len(rays_o)
            print_ok("decteced ", rays_sum, " rays within the thereshold ", threshold)            
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
                    # TODO: add sdf loss if necessary
                    # sdfs_point_rt = neus_to_aligin.generate_zero_sdf_points_with_RT(rays_o=rays_o_batch, rays_d=rays_d_batch, T=self.raw_translation, R=self.raw_quaternion) 
                    # # query those points' sdfs in standard neus
                    # neus_standard.sdf_network.sdf(sdfs_point_rt)
                    # mask_error = color_fine - rays_gt_batch
                    mask_error = (color_fine_01 - rays_gt_01_batch)
                    # mask_fine_loss = F.l1_loss(color_fine_01, torch.ones_like(color_fine_01),reduction='sum') / rays_sum / 3 # normalize within cnt 
                    mask_fine_loss = F.l1_loss(mask_error, torch.zeros_like(mask_error),reduction='sum') / rays_sum / 3.0 # normalize within cnt 
                    global_loss += mask_fine_loss.clone().detach()
                    mask_fine_loss.backward()  # img_loss for refine R & T
                    torch.cuda.synchronize()
                    # import pdb; pdb.set_trace()
                    
                    del render_out
                if write_out == "full" and vis_folder is not None:
                    # import pdb; pdb.set_trace();
                    debug_out_rgb = neus_to_aligin.render_novel_image_with_RTKM(q = self.raw_quaternion.detach().cpu().numpy(), t = self.raw_translation.detach().cpu().numpy(),
                            post_fix=image_index,  return_render_out=True, intrinsic_mat=neus_standard.dataset.intrinsics_all[image_index].detach().cpu().numpy(),
                            original_mat=orgin_mat_c2w.detach().cpu().numpy(), img_W=neus_to_aligin.dataset.W, img_H=neus_to_aligin.dataset.H, resolution_level=5)
                    # debug_path = vis_folder + "/" +  self.obj_names[to_aligin_index] + "_" + str(image_index) + ".png" # vis_folder is a path item
                    debug_path = vis_folder / str(iter_id) # vis_folder is a path item
                    if not debug_path.exists():
                        os.makedirs(debug_path)
                    cv.imwrite((str(debug_path) + "/" +  self.obj_names[to_aligin_index] + "_" + str(image_index) + ".png" ), debug_out_rgb)
                elif write_out == "fast" and vis_folder is not None: # fast means only render in the mask, same as genshin_nerf 
                    black_there_hold = 0
                    W, H, cnt_in_mask, cnt_in_rgb = self.W, self.H, 0, 0
                    debug_rgb = (np.concatenate(debug_rgb, axis=0).reshape(-1, 3) * 256).clip(0, 255).astype(np.uint8)
                    debug_img = np.zeros([H, W, 3]).astype(np.uint8)
                    for index in range(0, H):
                        for j in range(0, W):
                            if rays_mask[index][j]: # in the mask
                                if special_mask[cnt_in_mask]: # index cnt_in_mask also in the special mask
                                    if debug_rgb[cnt_in_rgb][0] > black_there_hold:
                                        debug_img[index][j][0] = debug_rgb[cnt_in_rgb][0] # store as in debug_image
                                        debug_img[index][j][1] = debug_rgb[cnt_in_rgb][1]
                                        debug_img[index][j][2] = debug_rgb[cnt_in_rgb][2]
                                    else: # write unfitted rgb as white
                                        debug_img[index][j][0] = 255
                                        debug_img[index][j][1] = 255
                                        debug_img[index][j][2] = 255
                                    cnt_in_rgb = cnt_in_rgb + 1 # move the index in debug rgb image to the nex one 
                                else: # outside the  special mask, highlighted as blue
                                        debug_img[index][j][0] = 0
                                        debug_img[index][j][1] = 255
                                        debug_img[index][j][2] = 0   
                                cnt_in_mask = cnt_in_mask + 1
                    print_blink("saving debug image at " + str(iter_id) + "th validation, with image inedex " + str(image_index))
                    cv.imwrite((vis_folder / (str(iter_id) + "_" + str(image_index) + ".png")).as_posix(), debug_img)
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
        # print_blink("R_ek_loss ", str(R_ek_loss.clone().detach().cpu().numpy()))
        R_ek_loss.backward()
        global_loss = R_ek_loss + global_loss
        return global_loss, visited
    
def get_optimizer(mode, honkaiStart):
    optimizer = None
    if  mode == "refine_rt":
        optimizer = torch.optim.Adam(
            [
                {'params': getattr(honkaiStart, 'raw_translation'), 'lr': 1e-3},
                {'params': getattr(honkaiStart, 'raw_quaternion'), 'lr': 1e-3},
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
            loss, visited = honkaiStart.refine_rt_forward(standard_index = honkaiStart.standard_index, to_aligin_index=honkaiStart.to_aligin_index, 
            vis_folder=vis_folder, write_out=write_out, iter_id = iter_id, single_image_refine=single_image_refine, images_total=1, start_index=start_index)
        else: # calc all images before refine
            loss, visited = honkaiStart.refine_rt_forward(standard_index = honkaiStart.standard_index, to_aligin_index=honkaiStart.to_aligin_index, 
                vis_folder=vis_folder, write_out=write_out, iter_id = iter_id)
        return loss, visited   
    optimizer = get_optimizer('refine_rt', honkaiStart=honkaiStart)
    train_iters = honkaiStart.train_iters
    pbar = trange(0, train_iters)
    img_len = len(honkaiStart.objects[honkaiStart.standard_index].dataset.images)
    for i in pbar:
        if single_image_refine: 
            for _ in range (0, single_sub_length):
                test_devision = 31
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
    parser.add_argument('--vis_folder', type=str, default="debug/dragon2to1_AABB3")
    parser.add_argument('--write_out', type=str, default=0)
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu) 
    honkaiStart = HonkaiStart(args.conf)    
    refine_rt(honkaiStart=honkaiStart, vis_folder= Path(args.vis_folder), single_image_refine=True, write_out=args.write_out)

"""
python reg.py --conf ./confs/json/march7th.json --gpu 1
python reg.py --conf ./confs/json/fuxuan.json --write_out fast --gpu 2 
python reg_AABB.py --conf ./confs/json/fuxuan.json --write_out fast --gpu 0
python reg.py --conf ./confs/json/klee.json --write_out fast --gpu 3
"""
    