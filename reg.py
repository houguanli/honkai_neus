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

class HonkaiStart(torch.nn.Module):
    def __init__(self, setting_json_path):
        super(HonkaiStart, self).__init__()
        self.device = 'cuda'
        with open(setting_json_path, "r") as json_file:
            reg_data = json.load(json_file)
        self.objects_cnt = reg_data['objects_cnt']
        self.train_iters = reg_data['train_iters']
        self.batch_size = reg_data['batch_size']
        self.raw_translation = torch.tensor(reg_data['raw_translation'], dtype=torch.float32, requires_grad=True) # this para is raw from bbox align
        self.raw_quaternion  = torch.tensor(reg_data['raw_quaternion'] , dtype=torch.float32, requires_grad=True) # 
        self.objects, self.obj_masks, self.obj_names = [], [], []
        for index in range (0, self.objects_cnt):
            current_name = str(index)
            current_obj_conf_path = reg_data['obj_confs'][current_name + "_conf"]
            current_obj_name = reg_data['obj_confs'][current_name + "_name"]
            current_exp_runner = Runner.get_runner(current_obj_conf_path, current_obj_name, is_continue=True)
            # pack this neus as a exp_runner in neus
            self.objects.append(current_exp_runner)
            current_mask = torch.sum(current_exp_runner.dataset.images, dim=-1)
            current_mask = (current_mask == 0) # [n_images, H, W, 3]
            self.obj_masks.append(current_mask)
            self.obj_names.append(current_obj_name)
        
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
        # return None, None
        sdf = self.runner_background.sdf_network.sdf(pts).contiguous()
        sdf_grad = self.runner_background.sdf_network.gradient(pts).squeeze().contiguous()
        return sdf, sdf_grad
    
    # TODO: need with torch.no_grad here? 
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
    
    def refine_rt_forward(self, standard_index = 0, to_aligin_index = 1, vis_folder=None, write_debug=False, iter_id = 0): # refine the rt from to_aligin_index to standard_index
        # raw_quad, raw_trans are set from the init or the optimizer
        # TODO select poses to make sure it is generate from a both-available area
        images_total = len(self.obj_masks[standard_index])
        print_info("running with ", images_total, "in total")
        global_loss = 0
        neus_standard, neus_to_aligin = self.objects[standard_index], self.objects[to_aligin_index]
        for image_index in range(0, images_total):
            # calc both equv z < 0, assumes in standard it is > 0, so only care about the equv in to_aligin_index
            orgin_mat_c2w = neus_standard.dataset.pose_all[image_index]
            pre_calc_c2w = self.calc_equivalent_camera_position(R = self.raw_quaternion, T = self.raw_translation, camera_c2w=orgin_mat_c2w)
            # print_info("calc eqv z ", pre_calc_c2w[2, 3])
            if pre_calc_c2w[2, 3] > 0: # in both avaible area, start reg
                print_info("Running at image ", image_index, "with equ z > 0")
                rays_o, rays_d, rays_gt = self.generate_samples(source_index=standard_index, image_index=image_index)
                rays_mask = self.obj_masks[standard_index][image_index]
                 # reshape is used for after mask, it become [rays_sum*3]
                rays_o, rays_d, rays_gt = rays_o[rays_mask].reshape(-1, 3), rays_d[rays_mask].reshape(-1, 3), rays_gt[rays_mask].reshape(-1, 3) 
                rays_sum = len(rays_o)  
                # TODO: reformat this function to suit for two poses rendering
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
                    threshold = 0.005
                    value = 1.0  # 大于阈值的元素将保持不变
                    value_below_threshold = 0.0  # 小于等于阈值的元素将置为0
                    color_fine_01 =  F.threshold(color_fine, threshold, value_below_threshold, value)
                    rays_gt_01_batch = F.threshold(rays_gt_batch, threshold, value_below_threshold, value)
                    # TODO: add sdf loss if necessary
                    mask_error = (color_fine_01 - rays_gt_01_batch)
                    mask_fine_loss = F.l1_loss(mask_error, torch.zeros_like(mask_error),reduction='sum') / rays_sum / 3 # normalize within cnt 
                    global_loss += mask_fine_loss.clone().detach()
                    mask_fine_loss.backward(retain_graph=True)  # img_loss for refine R & T
                    torch.cuda.synchronize()
                    del render_out
                if write_debug and vis_folder is not None:
                    # import pdb; pdb.set_trace();
                    debug_out_rgb = neus_to_aligin.render_novel_image_with_RTKM(q = self.raw_quaternion.detach().cpu().numpy(), t = self.raw_translation.detach().cpu().numpy(),
                            post_fix=image_index,  return_render_out=True, intrinsic_mat=neus_standard.dataset.intrinsics_all[image_index].detach().cpu().numpy(),
                            original_mat=orgin_mat_c2w.detach().cpu().numpy(), img_W=neus_to_aligin.dataset.W, img_H=neus_to_aligin.dataset.H, resolution_level=5)
                    # debug_path = vis_folder + "/" +  self.obj_names[to_aligin_index] + "_" + str(image_index) + ".png" # vis_folder is a path item
                    debug_path = vis_folder / str(iter_id) # vis_folder is a path item
                    if not debug_path.exists():
                        os.makedirs(debug_path)
                    cv.imwrite((str(debug_path) + "/" +  self.obj_names[to_aligin_index] + "_" + str(image_index) + ".png" ), debug_out_rgb)
            else:
                continue
            # count = 0
            print_info("acced loss at this index ", global_loss)
        R_ek_loss = torch.abs(torch.norm(self.raw_quaternion) - 1)
        print_blink("R_ek_loss ", str(R_ek_loss.clone().detach().cpu().numpy()))
        R_ek_loss.backward(retain_graph=True)
        global_loss = R_ek_loss + global_loss
        return global_loss
    
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

def refine_rt(honkaiStart : HonkaiStart, vis_folder=None): # runs as a train function 
    def refine_rt_forward(optimizer, iter_id=-1):
        optimizer.zero_grad()
        if vis_folder != None:
            if not os.path.exists(vis_folder):
                os.makedirs(vis_folder)
        loss = honkaiStart.refine_rt_forward(vis_folder=vis_folder, write_debug=True, iter_id = iter_id)
        return loss   
    optimizer = get_optimizer('refine_rt', honkaiStart=honkaiStart)
    train_iters = honkaiStart.train_iters
    pbar = trange(0, train_iters)
    for i in pbar:
        loss = refine_rt_forward(optimizer=optimizer, iter_id=i)
        if loss.norm() < 1e-3:
            break
        optimizer.step()    
        print("refining RAW_rt from pose 0 to pose 1")
        print('raw_translation: {}, raw_quaternion: {}, loss: {}'.format(honkaiStart.raw_translation, honkaiStart.raw_quaternion, loss.norm()))   
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
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu) 
    honkaiStart = HonkaiStart(args.conf)    
    refine_rt(honkaiStart=honkaiStart, vis_folder= Path("debug", "stand2lie"))

"""
python reg.py --conf ./confs/json/march7th.json --gpu 3
python reg.py --conf ./confs/json/fuxuan.json --gpu 0
python reg.py --conf ./confs/json/dragon.json --gpu 0
"""
    