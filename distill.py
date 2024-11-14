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
from pyhocon import ConfigFactory
from pathlib import Path
import os
from models.fields import * 
from models.renderer import extract_geometry, sample_pdf, NeuSRenderer

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
        self.raw_translation = torch.tensor(reg_data['raw_translation'], dtype=torch.float32) # this para is raw from bbox align
        self.raw_quaternion  = torch.tensor(reg_data['raw_quaternion'] , dtype=torch.float32)

        self.objects, self.obj_masks, self.obj_names, self.frnns, self.zero_sdf_points_all, self.zero_sdf_points_all_mask = [], [], [], [], [], [] # all arrays have the number of objects 
        for index in range (0, self.objects_cnt):
            current_name = str(index)
            current_obj_conf_path = reg_data['obj_confs'][current_name + "_conf"]
            current_obj_name = reg_data['obj_confs'][current_name + "_name"]
            current_exp_runner = Runner.get_runner(current_obj_conf_path, current_obj_name, is_continue=True)
            # pack this neus as a exp_runner in neus
            self.objects.append(current_exp_runner)
            current_sum = torch.sum(current_exp_runner.dataset.images, dim=-1)
            current_mask = (current_sum > 0.02)
            self.obj_masks.append(current_mask)
            self.obj_names.append(current_obj_name)
        self.W, self.H = self.obj_masks[0].shape[2], self.obj_masks[0].shape[1] # notice the index, self.obj_masks contains N sets of masks
        self.source_images_total, self.target_images_total = len(self.obj_masks[self.source_index]), len(self.obj_masks[self.target_index])

        # Dstill
        # create descendant network      
        params_to_train = []
        self.conf_path = reg_data['network_conf']
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case_name)
        f.close()
        self.conf = ConfigFactory.parse_string(conf_text)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # self.student_nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        # self.student_sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        # self.student_deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        # self.student_color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        self.student_nerf_outside = self.objects[self.target_index].nerf_outside
        self.student_sdf_network = self.objects[self.target_index].sdf_network
        self.student_deviation_network = self.objects[self.target_index].deviation_network
        self.student_color_network = self.objects[self.target_index].color_network
        params_to_train += list(self.student_nerf_outside.parameters())
        params_to_train += list(self.student_sdf_network.parameters())
        params_to_train += list(self.student_deviation_network.parameters())
        params_to_train += list(self.student_color_network.parameters())
        
        self.renderer = NeuSRenderer(self.student_nerf_outside,
                                     self.student_sdf_network,
                                     self.student_deviation_network,
                                     self.student_color_network,
                                     **self.conf['model.neus_renderer'])

        self.is_continue = is_continue
        
        self.iter_step = 0
        self.end_iter = self.conf.get_int('train.end_iter')

        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.base_exp_dir = Path('exp') / 'distill' / case_name
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
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
            'nerf': self.student_nerf_outside.state_dict(),
            'sdf_network_fine': self.student_sdf_network.state_dict(),
            'variance_network_fine': self.student_deviation_network.state_dict(),
            'color_network_fine': self.student_color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint,
                   os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name),
                                map_location=self.device)
        self.student_nerf_outside.load_state_dict(checkpoint['nerf'])
        self.student_sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.student_deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.student_color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

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

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

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
        iter_i = 0
        for iter_i in tqdm.tqdm(range(res_step)):
            if iter_i % 2 == 0: # swtich the teacher neus for each step
                # Using source NeuS as teacher
                image_perm = torch.randperm(self.source_images_total)
                image_index = image_perm[(self.iter_step // 2) % len(image_perm)]
                neus_index = self.source_index
            else:
                # Using target Neus as teacher
                image_perm = torch.randperm(self.target_images_total)
                image_index = image_perm[(self.iter_step // 2) % len(image_perm)]
                neus_index = self.target_index

            sample_sphere_raidus = 0.55
            data = self.objects[neus_index].dataset.gen_random_rays_at(image_index, self.batch_size)
            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
            # No matter which neus is the teacher, the sample points are always generated in the target neus space
            # Thus we transform the sample points to the target neus space if the teacher is the source neus
            if neus_index == self.source_index:
                transfrom_matrix, transfrom_matrix_inv = self.get_transform_matrix(translation=self.raw_translation, quaternion=self.raw_quaternion)
                rays_o = (transfrom_matrix[None, :3, :3] @ rays_o[:, :, None]).squeeze(dim=-1)
                rays_d = (transfrom_matrix[None, :3, :3] @ rays_d[:, :, None]).squeeze(dim=-1)
                T1_expand = (transfrom_matrix[0:3, 3]).repeat(len(rays_o), 1)
                rays_o = rays_o + T1_expand
                rays_d = rays_d + T1_expand

            near, far = self.objects[neus_index].dataset.near_far_from_sphere(rays_o, rays_d, radius=sample_sphere_raidus)

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)
            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.distill_render(rays_o, rays_d, near, far, 
                                            transfrom_matrix_inv, self.objects[self.source_index], self.objects[self.target_index],
                                            background_rgb=background_rgb,
                                            cos_anneal_ratio=self.get_cos_anneal_ratio())
            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            # sdf_loss = render_out['sdf_loss']
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

            loss =  color_fine_loss + \
                    eikonal_loss * self.igr_weight + \
                    mask_loss * self.mask_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.iter_step += 1
            iter_i += 1
            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))
            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()
            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_student_mesh()
            if self.iter_step % self.val_freq == 0:
                self.validate_image()
            self.update_learning_rate()
    
    def validate_image(self, idx=-1, resolution_level=-1):
        # randomly selete 0 or 1 
        neus_index = np.random.randint(2)
        if neus_index == self.source_index:
            idx = np.random.randint(self.source_images_total)
            print_info('validate in source datasets..')
        else:
            idx = np.random.randint(self.target_images_total)
            print_info('validate in target datasets..')
        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))
        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.objects[neus_index].dataset.gen_rays_at(idx, resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        if neus_index == self.source_index:
            transfrom_matrix, _ = self.get_transform_matrix(translation=self.raw_translation, quaternion=self.raw_quaternion)
            rays_o = (transfrom_matrix[None, :3, :3] @ rays_o[:, :, None]).squeeze(dim=-1)
            rays_d = (transfrom_matrix[None, :3, :3] @ rays_d[:, :, None]).squeeze(dim=-1)
            T1_expand = (transfrom_matrix[0:3, 3]).repeat(len(rays_o), 1)
            rays_o = rays_o + T1_expand
            rays_d = rays_d + T1_expand

        out_rgb_fine = []
        out_normal_fine = []
        depth_map = []
        for rays_o_batch, rays_d_batch in zip(rays_o.split(self.batch_size), rays_d.split(self.batch_size)):
            near, far = self.objects[neus_index].dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
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

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.objects[neus_index].dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                            .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                            np.concatenate([img_fine[..., i],
                                            self.objects[neus_index].dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                            normal_img[..., i])

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
    if args.mode == "distill":
        honkaiStart.train_student_sdf(is_continue = args.is_continue)
    elif args.mode == "validate_mesh":
        honkaiStart.validate_student_mesh(is_continue = args.is_continue, resolution=512)
    elif args.mode == "validate_image":
        honkaiStart.validate_image()
    seed = 20031012
    

"""
python reg_FRNN.py --conf ./confs/json/fuxuan.json --write_out fast --gpu 2 
python reg_FRNN.py --conf ./confs/json/klee.json --write_out fast --gpu 3
python reg_FRNN.py --conf ./confs/json/fuxuan_fricp.json --write_out fast --gpu 0
python reg_FRNN.py --conf ./confs/json/fuxuan_fricp.json --mode render_ply --write_out fast --gpu 1
python reg_FRNN.py --conf ./confs/json/fuxuan_fricp.json --mode distill --gpu 1  --is_continue
python reg_FRNN.py --conf ./confs/json/fuxuan_fricp.json --mode validate --gpu 1 --is_continue
"""
    