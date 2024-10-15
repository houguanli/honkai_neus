import os
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer
from models.common import * 
import json
import open3d as o3d
from pathlib import Path

class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()
        self.name = case
        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'])

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

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()

        for iter_i in tqdm(range(res_step)):
            data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)
            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
            # center = torch.Tensor([0.05, -0.1, 0]).cuda()
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.zeros([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(rays_o, rays_d, near, far,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())

            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']

            # Loss
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss = gradient_error

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            loss = color_fine_loss + \
                   eikonal_loss * self.igr_weight + \
                   mask_loss * self.mask_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def train_dynamic(self):
        # TODO use render_dynamic to pass img_loss
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'dynamic_logs'))
        return

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name),
                                map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint,
                   os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)
        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))
        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []
        depth_map = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.zeros([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)
            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            if feasible('depth_map'):
                depth_map.append(render_out['depth_map'].detach().cpu().numpy())
            del render_out
        # depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
        # import pdb; pdb.set_trace()
        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        if len(depth_map) > 0:
            depth_map = (np.concatenate(depth_map, axis=0).reshape([H, W, -1]) * 256).clip(0, 255).astype(np.uint8)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(depth_map) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}_depth.png'.format(self.iter_step, i, idx)),
                           depth_map[..., i])
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.zeros([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def render_novel_image_at(self, camera_pose, resolution_level, intrinsic_inv=None):
        rays_o, rays_d = self.dataset.gen_rays_at_pose_mat(camera_pose, resolution_level=resolution_level,intrinsic_inv=intrinsic_inv)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        out_rgb_fine = []
        normal_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.zeros([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)
            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            del render_out
        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine, normal_fine
    
    def render_depth_image_at(self, camera_pose, resolution_level, intrinsic_inv=None): # TODO: to be finished
        rays_o, rays_d = self.dataset.gen_rays_at_pose_mat(camera_pose, resolution_level=resolution_level,intrinsic_inv=intrinsic_inv)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        out_depth_fine = []
        return 
    
    def query_points_sdf(self, query_points):
        return self.sdf_network.sdf(query_points).contiguous().squeeze()
    
    # this function split zero  sdf points from all zero sdf points 
    def split_zero_sdf_points(self, zero_points_all):
        start_index, zero_sdf_points_all = 0, []
        for image_index in range(0, len(self.dataset.images)):
            mask = self.dataset.masks[image_index] > 0 # notice that the image is pre masked! 
            mask_len = (int)(mask.sum() / 3)
            zero_sdf_points_all.append(zero_points_all[start_index : start_index + mask_len, :])
            start_index = start_index + mask_len 
        # import pdb; pdb.set_trace()
        
        return zero_sdf_points_all
    
    # this function generate_zero_sdf_points (N, 3) from rays_o rays_d.
    def generate_zero_sdf_points(self, rays_o, rays_d, zero_sdf_thereshold=1e-3, inf_depth_thereshold=2.0, return_all = True):
        depths, sdfs = torch.zeros(len(rays_o), dtype=torch.float32), torch.ones(len(rays_o), dtype=torch.float32)
        rays_mask, zero_mask, inf_mask =  torch.ones((len(rays_o)), dtype=torch.bool), \
            torch.ones((len(rays_o)), dtype=torch.bool), torch.ones((len(rays_o)), dtype=torch.bool) 
        rays_o = rays_o.clone() # VERY important
        # this is a calculation using progressive photon mapping to calculate depth for each single ray, upper is its refinement , 
        # using batch calculation to accelerate the program with only one while logic term.original code please to refer to 
        # genshin_nerf render_depth_core. Now depth is the points from rays_o to surface as the dir is rays_d. however, need to remove outer ones       
        while torch.sum(rays_mask) > 0:
            pts, dirs = rays_o[rays_mask], rays_d[rays_mask]
            tmp_sdfs =  self.sdf_network.sdf(pts).contiguous().squeeze()
            pts = pts + dirs * (tmp_sdfs.repeat(3, 1).T)
            rays_o[rays_mask] = pts # update current_rays
            depths[rays_mask] = depths[rays_mask] + tmp_sdfs
            sdfs[rays_mask] = tmp_sdfs
            zero_mask, inf_mask = sdfs < zero_sdf_thereshold, sdfs > inf_depth_thereshold #must be sdfs be
            rays_mask = zero_mask + inf_mask
            rays_mask = ~rays_mask
        # print(sdfs)
        zero_points = None
        if return_all:
            zero_points = rays_o
        elif torch.sum(zero_mask) > 0: # have legeal depth value
            zero_points = rays_o[zero_mask] # below is wrong because rayso is updated ! 
            # zero_points = rays_o[zero_mask] + rays_d[zero_mask] * (depths[zero_mask].repeat(3, 1).T)
            # # only consider the points that reaching the zero thereshold 
        else:
            print("No points are generated from SDF field")
        return zero_points, zero_mask      # note it returns a torch tensor
    
    # this function is used for out calling 
    def generate_zero_sdf_points_with_RT(self, rays_o, rays_d, q, t, zero_sdf_thereshold=1e-3, inf_depth_thereshold=2.0): 
        # expand as 4x4mat 
        w, x, y, z = q
        rotate_mat = torch.zeros((3, 3), device=rays_o.device)
        rotate_mat[0, 0] = 1 - 2 * (y ** 2 + z ** 2)
        rotate_mat[0, 1] = 2 * (x * y - z * w)
        rotate_mat[0, 2] = 2 * (x * z + y * w)
        rotate_mat[1, 0] = 2 * (x * y + z * w)
        rotate_mat[1, 1] = 1 - 2 * (x ** 2 + z ** 2)
        rotate_mat[1, 2] = 2 * (y * z - x * w)
        rotate_mat[2, 0] = 2 * (x * z - y * w)
        rotate_mat[2, 1] = 2 * (y * z + x * w)
        rotate_mat[2, 2] = 1 - 2 * (x ** 2 + y ** 2)
        transform_matrix = torch.zeros((4, 4), device=rays_o.device)
        transform_matrix[0:3, 0:3] = rotate_mat
        transform_matrix[0:3, 3] = t
        transform_matrix[3, 3] = 1.0
        # rotate first
        rays_o = torch.matmul(transform_matrix[None, :3, :3], rays_o[:, :, None]).squeeze(dim=-1)  # W, H, 3
        rays_o = rays_o + transform_matrix[None, :3, 3]
        rays_d = torch.matmul(transform_matrix[None, :3, :3], rays_d[:, :, None]).squeeze(dim=-1)  # W, H, 3
        # print(rays_d.shape)
        # rays_o = torch.matmul(camera_pos[None, :3, :3], rays_o[:, :, None]).squeeze()  # batch_size, 3
        # rays_o = rays_o + T1_expand  # batch_size 3, R1*T0 + T1
        # print("equivalent c2w mat: \n", camera_pos.clone().detach().cpu())
        return self.generate_zero_sdf_points(rays_o=rays_o, rays_d=rays_d,
                                             zero_sdf_thereshold=zero_sdf_thereshold, inf_depth_thereshold=inf_depth_thereshold)
    
    def validate_mesh(self, world_space=False, resolution=256, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles = \
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)), encoding='ascii')
        print("save at " + os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))
        logging.info('End')

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                                                  resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,'{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),fourcc, 30, (w, h))
        for image in images:
            writer.write(image)
        writer.release()
   
    # this function generate zero-sdf points of one training image, and save it as a ply file if write_out_flag
    # returns np array, and only focus the points IN the mask
    def generate_and_save_points_ply_single(self, store_path=None, image_index=0, resolution_level=1, write_out_flag=True):
        if store_path is None:
            store_dir = Path("debug", "zero_points_test")
            if not os.path.exists(store_dir):
                os.makedirs(store_dir)
            store_path = str(store_dir) + "/" + self.name + "_" + str(image_index) + ".ply"
        camera_c2w, intrinsic_inv = self.dataset.pose_all[image_index].clone(), self.dataset.intrinsics_all_inv[image_index].clone()
        rays_o, rays_d = self.dataset.gen_rays_at_pose_mat(camera_c2w, resolution_level=resolution_level,intrinsic_inv=intrinsic_inv, is_np=False)
        # only bactchfy the rays in the mask
        rays_mask = self.dataset.masks[image_index] > 0
        rays_o, rays_d = rays_o[rays_mask], rays_d[rays_mask]
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        rays_sum, process_flag = len(rays_o), 0
        zero_points_all, zero_mask_all = np.empty((0,3)), np.empty((0,1))
        for rays_o_batch, rays_d_batch in zip(rays_o.split(self.batch_size), rays_d.split(self.batch_size)):
            # with torch.no_grad():
            zero_points, zero_mask = self.generate_zero_sdf_points(rays_o=rays_o_batch, rays_d=rays_d_batch)
            if zero_points is None:
                continue
            zero_points = zero_points.detach().cpu().numpy()
            zero_mask = zero_mask.detach().cpu().numpy()
            zero_mask = zero_mask.reshape(-1, 1)
            # import pdb; pdb.set_trace()
            
            zero_points_all = np.concatenate((zero_points_all, zero_points), axis=0) # contact results
            zero_mask_all   = np.concatenate((zero_mask_all, zero_mask), axis=0)  # contact mask results
            del zero_points, zero_mask
            process_flag += self.batch_size
            # print("calculating rays ", process_flag, "with total ", rays_sum)
        if write_out_flag:
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(zero_points_all)
            o3d.io.write_point_cloud(store_path, point_cloud)
        return zero_points_all, zero_mask_all
    
    # this function generate ALL zero-sdf points of ALL training image, and save it as a ply file if write_out_flag
    # returns np array
    def generate_zero_points_full(self, store_dir=None, save_per_image=False, write_out_flag=True):
        if store_dir is None:
            store_dir = store_dir = Path("debug", "zero_points_test")
        if not os.path.exists(store_dir):
            os.makedirs(store_dir)
        singe_points_full, zero_sdf_points_all = np.empty((0, 3)), []
        for image_index in range(0, len(self.dataset.images)):
            store_path = str(store_dir) + "/" + self.name + "_" + str(image_index) + ".ply"
            # generate this for every image
            singe_points_full_single_image, _ = \
                self.generate_and_save_points_ply_single(store_path, image_index=image_index, resolution_level=1, write_out_flag=save_per_image) 
            singe_points_full = np.concatenate((singe_points_full, singe_points_full_single_image), axis=0) # contact results
            zero_sdf_points_all.append(singe_points_full_single_image)
            print_blink("concatenated points generated from image " + str(image_index))
        if write_out_flag:
            point_cloud = o3d.geometry.PointCloud() # auto write out
            store_path = str(store_dir) + "/" + self.name + "_full.ply"
            point_cloud.points = o3d.utility.Vector3dVector(singe_points_full)
            o3d.io.write_point_cloud(store_path, point_cloud)
        return zero_sdf_points_all
   
    def generate_zero_points_full_with_masks(self, store_dir=None, save_per_image=False, write_out_flag=True):
        if store_dir is None:
            store_dir = store_dir = Path("debug", "zero_points_test")
        if not os.path.exists(store_dir):
            os.makedirs(store_dir)
        singe_points_full, zero_sdf_points_all, points_mask_full = np.empty((0, 3)), [], np.empty((0, 1))
        for image_index in range(0, len(self.dataset.images)):
            store_path = str(store_dir) + "/" + self.name + "_" + str(image_index) + ".ply"
            # generate this for every image
            singe_points_full_single_image, points_mask = \
                self.generate_and_save_points_ply_single(store_path, image_index=image_index, resolution_level=1, write_out_flag=save_per_image) 
            singe_points_full = np.concatenate((singe_points_full, singe_points_full_single_image), axis=0) # contact results
            zero_sdf_points_all.append(singe_points_full_single_image)
            points_mask_full = np.concatenate((points_mask_full, points_mask), axis=0) # contact results
            print_blink("concatenated points generated from image " + str(image_index))
        if write_out_flag:
            point_cloud = o3d.geometry.PointCloud() # auto write out
            store_path = str(store_dir) + "/" + self.name + "_full.ply"
            point_cloud.points = o3d.utility.Vector3dVector(singe_points_full)
            o3d.io.write_point_cloud(store_path, point_cloud)
            store_path = str(store_dir) + "/" + self.name + "_full_mask.txt"
            np.savetxt(store_path, points_mask_full, fmt='%d')
        return zero_sdf_points_all, points_mask_full
        
    def render_novel_image_with_RTKM(self, post_fix=1, original_mat=None, intrinsic_mat=None, q=None, t=None,
                                     img_W=800, img_H=600, return_render_out=False, resolution_level=1):
        if q is None or t is None:
            q, t = [1, 0, 0, 0], [0, 0, 0] # this is a default setting
            # q = [0, 0, 1, 0] 
            # t = [0, -0.01, 0.066] # soap1 pose to soap2 pose 
            # q = [0.9150, -0.2691, -0.1273,  0.2763]
            # t = [0.1536, -0.1478,  0.3126]  # frame 0 qt calced from soap2 pose (default), IMP: pose2 is the default rendering pose !
            # q = [0.12746657701903685 , -0.27666154933511306 , 0.9138042514674574 , -0.26945212785406775]
            # t = [0.13278135983999997 , -0.12696580667999996 , 0.3725301366] # eqv rt for soap1 pose   
            # q = [ 0.6053165197372437, 0.2681955397129059, -0.37045902013778687, 0.6537007689476013]
            # t = [ 0.24793949723243713, 0.6238101124763489, 0.677591860294342] # 20th frame qt for pose2
            q, t = [0.1830127090215683,-0.6830127239227295,-0.1830127090215683,-0.6830127239227295],[0.1000, 0.40000, 0.25],
        w, x, y, z = q
        if original_mat is None:
            original_mat = np.array(
        [
            [
                1.0,
                -0.0,
                -0.0,
                -0.10000000149011612
            ],
            [
                0.0,
                -0.9929816722869873,
                0.1182689294219017,
                0.09290292859077454
            ],
            [
                0.0,
                -0.1182689294219017,
                -0.992981493473053,
                1.11918306350708
            ],
            [
                0.0,
                -0.0,
                -0.0,
                1.0
            ]
        ]
        )
        if intrinsic_mat is None:
            intrinsic_mat = np.array(
        [
            [
                1111.1110311937682,
                0.0,
                400.0
            ],
            [
                0.0,
                1111.1110311937682,
                300.0
            ],
            [
                0.0,
                0.0,
                1.0
            ]
        ]
            )       
        rotate_mat = np.array([
            [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
        ])
        transform_matrix = np.zeros((4, 4))
        transform_matrix[0:3, 0:3] = rotate_mat
        transform_matrix[0:3, 3] = t
        transform_matrix[3, 3] = 1.0
        # inverse_matrix = np.linalg.inv(transform_matrix)
        intrinsic_inv = torch.from_numpy(np.linalg.inv(intrinsic_mat).astype(np.float32)).cuda()
        camera_pose = np.array(original_mat)
        transform_matrix = transform_matrix @ camera_pose
        # transform_matrix = np.array([
        #     [0.0433,   0.0397, -0.9953,  0.8153],
        #     [ 0.9944,  -0.0608,  0.0418,  0.1112],
        #     [-0.0613, -0.9963, -0.0445,  0.1639],
        #     [ 0.0,          0.0,          0.0, 1]
        # ]) # tmp
        self.dataset.W = img_W
        self.dataset.H = img_H
        # print("equivalent c2w mat: \n", transform_matrix)
        # transform_matrix =transform_matrix.astype(np.float32).cuda()
        img, normal = self.render_novel_image_at(transform_matrix, resolution_level=resolution_level, intrinsic_inv=intrinsic_inv)
        # img loss
        # set_dir, file_name_with_extension = os.path.dirname(setting_json_path), os.path.basename(setting_json_path)
        # file_name_with_extension = os.path.basename(setting_json_path)
        # case_name, file_extension = os.path.splitext(file_name_with_extension)
        render_path = os.path.join(self.base_exp_dir, "test_" + str(post_fix) + ".png")
        if return_render_out:
            return img
        else:
            cv.imwrite(render_path, img)
            print("Saving render img at " + render_path)
            return None

    def get_runner(neus_conf_path, case_name, is_continue):
        return Runner(neus_conf_path, mode="train", case=case_name, is_continue=is_continue)


if __name__ == '__main__':
    print('Honkai NerF, start!!!')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # torch.cuda.set_device(args.gpu)
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--post_fix', type=str, default='0')
    parser.add_argument('--resolution_level', type=int, default=1)
    parser.add_argument('--store_dir', type=str, default=None)
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu) 
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=False, resolution=512, threshold=args.mcube_threshold)
    elif args.mode == 'validate_image':
        runner.validate_image()
    elif args.mode == 'generate_points':
        runner.generate_zero_points_full(store_dir=args.store_dir)
    elif args.mode == 'generate_points_wm':
        runner.generate_zero_points_full_with_masks(store_dir=args.store_dir)
    elif args.mode == 'render_rtkm':
        runner.render_novel_image_with_RTKM(post_fix=args.post_fix, resolution_level=args.resolution_level)
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)

"""
conda activate neus
cd D:/gitwork/NeuS
python exp_runner.py --mode validate_mesh --conf ./confs/thin_structure.conf --case scene1 --is_continue --gpu 4
python exp_runner.py --mode validate_image --conf ./confs/thin_structure_white_bkgd.conf --case soap2_merge --is_continue --gpu 5
python exp_runner.py --mode render_rtkm --conf ./confs/thin_structure_white_bkgd.conf --case soap2_merge --is_continue --gpu 5
python exp_runner.py --mode train --conf ./confs/thin_structure_white_bkgd.conf --case bunny2
python exp_runner.py --mode render_rtkm --conf ./confs/thin_structure_white_bkgd.conf --is_continue --gpu 0 --case tree
python exp_runner.py --mode generate_points --conf ./confs/thin_structure_white_bkgd.conf --is_continue --gpu 2 --case bunny_stand
python exp_runner.py --mode generate_points --conf ./confs/thin_structure_white_bkgd.conf --is_continue --gpu 3 --case dragon_pos1 --store_dir ./exp/dragon_pos1
python exp_runner.py --mode generate_points --conf ./confs/thin_structure_white_bkgd.conf --is_continue --gpu 0 --case dragon_pos2 --store_dir ./exp/dragon_pos2
python exp_runner.py --mode generate_points_wm --conf ./confs/thin_structure_white_bkgd.conf --is_continue --gpu 0 --case dragon_pos2 --store_dir ./exp/dragon_pos2
"""
