import json
import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as F
from models.common import *
from argparse import ArgumentParser
from pyhocon import ConfigFactory
import trimesh
import open3d as o3d
from pathlib import Path
import os
import argparse
from models.fields import SDFNetwork
from models.dataset import load_K_Rt_from_P
from models.renderer import extract_geometry
import models.py_fricp as py_fricp

# The registration process is divided into two steps: coarse registration and fine registration
# Keep in mind we always transfrom the source neus to the target neus, which means the restult of the registration is the transformation matrix from source to target
class NeuSReg:
    target_neus = None
    source_neus = []
    target_neus = None
    source_mesh = []
    source_num = 0

    # each source neus has to be registered to target neus, and the target neus is unique.
    coarse_tranform = []
    fine_transform = []
    source_to_target = []
    fricp = py_fricp.PY_FRICPd()

    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config

        neus_config_path = config['neus_config']
        self.source_num = len(config['source_neus'])
        f = open(neus_config_path)
        conf_text = f.read()
        f.close()
        self.neus_config = ConfigFactory.parse_string(conf_text)

        self.target_sdf_net = SDFNetwork(**self.neus_config['model.sdf_network']).to(self.device)
        target_checkpoint = torch.load(config['target_ckpt'], map_location=self.device)
        self.target_sdf_net.load_state_dict(target_checkpoint['sdf_network_fine'])
        self.target_mesh = self.validate_mesh(self.target_sdf_net)
        print_info('load source sdf network from', config['source_ckpt'])

        for i in range(self.source_num):
            print_info('load target sdf network from', config['source_ckpt'][i])
            self.source_neus.append(SDFNetwork(**self.neus_config['model.sdf_network'][i]).to(self.device))
            source_checkpoint = torch.load(config['source_ckpt'][i], map_location=self.device)
            self.source_neus[i].load_state_dict(source_checkpoint['sdf_network_fine'])

            # initialize the transformation matrix
            self.source_mesh.append(self.validate_mesh(self.source_neus[i]))
        print_info(f'Load {self.source_num} target sdf networks in total!')

    def validate_mesh(self, sdf_network):
        bound_min = torch.tensor([-1.0, -1.0, -1.0, 1.0], dtype=torch.float32)
        bound_max = torch.tensor([ 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        resolution = 256
        threshold = 0.0

        # we first extract a coarse mesh to get the bounding box, and then extract a fine mesh
        vertices, triangles = \
            extract_geometry(bound_min,
                            bound_max,
                            resolution=resolution,
                            threshold=threshold,
                            query_func=lambda pts: -sdf_network.sdf(pts))

        mesh = trimesh.Trimesh(vertices, triangles)
        bound_min, bound_max = mesh.bounds

        # extract fine mesh
        resolution = 512
        vertices, triangles = \
            extract_geometry(bound_min,
                            bound_max,
                            resolution=resolution,
                            threshold=threshold,
                            query_func=lambda pts: -sdf_network.sdf(pts))
        
        mesh = trimesh.Trimesh(vertices, triangles)
        # export mesh
        return mesh

    # merge the source and target dataset according to the registration result
    # Specifically, we need to transform the source camera extrinsic to the target camera spatial coordinate
    def merge_dataset(self):
        if self.config['new_case'] == '':
            print_error('Please specify the new case name!')
            return
        new_case_name = self.config['new_case']
        target_case = self.neus_config['target_case']
        target_camera_dict = np.load(os.path.join('./public_data', target_case, 'cameras_sphere.npz'))
        target_image_list = os.listdir(os.path.join('./public_data', target_case, 'images'))
        target_image_num = len(target_image_list)
        # copy the target dataset to the new case
        os.system('cp -r %s %s' % (os.path.join('./public_data', target_case), os.path.join('./public_data', new_case_name)))
        print_info(f'Finish copying the target dataset to the new case {new_case_name}!')

        for i in range(self.source_num):
            source_case = self.neus_config['source_case'][i]
            # load the cameras_sphere.npz file of the source and target dataset
            source_camera_dict = np.load(os.path.join('./public_data', source_case, 'cameras_sphere.npz'))
            source_image_list = os.listdir(os.path.join('./public_data', source_case, 'images'))
            source_image_num = len(source_image_list)
            # world_mat is a projection matrix from world to image
            self.world_mats_np = [source_camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(source_image_num)]
            # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
            self.scale_mats_np = [source_camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(source_image_num)]
            count = 0

            for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
                P = world_mat @ scale_mat
                P = P[:3, :4]
                intrinsic, pose = load_K_Rt_from_P(None, P)
                new_pos = self.source_to_target[i] @ pose
                intrinsic44 = np.eye(4, dtype=np.float32)
                intrinsic44[:3, :3] = intrinsic

                # update the world_mat
                world_mat = intrinsic44 @ np.linalg.inv(new_pos)
                world_mat = world_mat.astype(np.float32)
                # update the camera_dict
                target_camera_dict['world_mat_%d' % (target_image_num + count * i)] = world_mat
                count += 1
                # copy the source image to the target image folder
                source_image_path = os.path.join('./public_data', source_case, 'image', f'{target_image_num + count * i:03d}.png')
                target_image_path = os.path.join('./public_data', new_case_name, 'image', f'{target_image_num + count * i:03d}.png')
                os.system('cp %s %s' % (source_image_path, target_image_path))
                source_mask_path = source_image_path.replace('image', 'mask')
                target_mask_path = target_image_path.replace('image', 'mask')
                os.system('cp %s %s' % (source_mask_path, target_mask_path))
            print_info(f'Finish merging source {i} dataset to target dataset!')
        print_info(f'Finish merging all source datasets to target dataset!')
        
    def coarse_registration(self):
        # calculate the source mesh's obb 
        target_obb = self.target_mesh.bounding_box_oriented
        for i in range(self.source_num):
            pass
        
    def fine_registration(self):
        target_points = np.asarray(self.target_neus.vertices)
        for i in range(self.source_num):
            source_points = torch.from_numpy(np.asarray(self.source_mesh[i].vertices))
            # apply the coarse transformation to the source points
            source_points = self.coarse_tranform[i][None, :3, :3] @ source_points[:, :, None]
            T_expand = (self.coarse_tranform[i][0:3, 3]).repeat(len(source_points), 1)
            source_points = source_points + T_expand

            # convert to numpy array
            source_points = source_points.cpu().numpy()
            self.fricp.set_points(source_point=source_points, target_point=target_points)
            self.fine_transform[i] = self.fricp.run_icp(method=3)

            self.source_to_target[i] = self.fine_transform[i] @ self.coarse_tranform[i]
            print_info(f'fine registration for target {i} is done!')
    
    def run_registration(self):
        self.coarse_registration()
        self.fine_registration()
    
    def blend_neus(self):
        self.merge_dataset()
        os.system(f'python exp_runner.py --config {self.neus_config} --case {self.config['new_case']} --mode train')

argparser = argparse.ArgumentParser()
argparser.add_argument('--mode', type=str, default='reg')
argparser.add_argument('--config', type=str, default='configs/hongkai_neus.json')
argparser.add_argument('--gpu', type=int, default=0)
argparser.add_argument('--blend', action='store_true', default=False)

if __name__ == '__main__':
    args = argparser.parse_args()
    torch.set_default_device(args.gpu)
    with open(args.config, 'r') as f:
        config = json.load(f)

    neus_reg = NeuSReg(config)
    mode = args.mode
    if mode == 'reg':
        neus_reg.run_registration()
    if args.blend:
        neus_reg.blend_neus()