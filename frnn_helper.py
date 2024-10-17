import frnn
import torch
import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
import torch


dim = 3
max_length = 200
init_neighbor_num = 1
init_query_point_num = 514

class FRNN:

    def __init__(self, npz_file_path, point_mask_path=None):
        # run a random query to generate grid
        data = o3d.io.read_point_cloud(npz_file_path)
        self.points = np.asarray(data.points, dtype=np.float32)
        self.unmasked_points = self.points # store the unmasked_points if it will be used
        self.frnn_grid, self.mask = None, None
        if point_mask_path is not None:
            self.mask = np.loadtxt(point_mask_path, dtype=bool)
            point_cloud = o3d.geometry.PointCloud() # auto write out
            self.points = self.points[self.mask] # remove outlier
            # store_path ="./debug/test_mask.ply"
            # point_cloud.points = o3d.utility.Vector3dVector(self.points)
            # o3d.io.write_point_cloud(store_path, point_cloud)
        self.build_frnn_on_torch()
        return
    
    def build_frnn_on_torch(self):
        self.points = torch.from_numpy(self.points).to('cuda')
        if self.mask is not None: # must be numpy
            self.mask = torch.from_numpy(self.mask).to('cuda')
            self.unmasked_points = torch.from_numpy(self.unmasked_points).to('cuda')
        test = 114 * torch.rand(init_query_point_num, 3).cuda()
        # reshape for initial query 
        # N, P1, dim
        # N: Num of Point cloud
        # P1: How many points are contained in point clouds
        # dim: spatial dimension
        # source_points = points, differ in shape
        self.source_points = self.points.reshape(1, len(self.points), dim) # change the shape
        test = test.reshape(1, init_query_point_num, dim)
        _, _, _, grid = frnn.frnn_grid_points(
        test, self.source_points, None, None, init_neighbor_num, max_length,\
            grid=None, return_nn=False, return_sorted=True)
        self.grid = grid # save the grid for further query
        
    def query_Knear_points(self, query_points, points_num): # torch, N, 3
        query_points = query_points.reshape(1, len(query_points), dim)
        dists, idxs, nn, _ = frnn.frnn_grid_points(
        query_points, self.source_points, None, None, points_num, max_length,\
            grid=self.grid, return_nn=False, return_sorted=True)
        return dists, idxs, nn
    
    def query_nearest_points(self, query_points):# torch, N, 3, only return the nearest
        dists, idxs, _ = self.query_Knear_points(query_points, 1) # K = 1
        return dists.squeeze(), idxs.squeeze() # return shape N N 
    
    def judge_points_within_ball(self, query_points, ball_dist, max_init_points = max_length, threshold_cnt=1): # default is the nearest
        dists, _, _ = self.query_Knear_points(query_points, max_init_points) # K = max len
        dists = dists.squeeze()
        ava_flag = dists <= ball_dist
        ava_flag = torch.sum(ava_flag, dim=1) >= threshold_cnt
        # import pdb; pdb.set_trace()
        return ava_flag
    
    def get_frnn_tree(npz_path, point_mask_path=None):
        return FRNN(npz_path,point_mask_path=point_mask_path)  
        
"""
# dists: distance
# idxs: index(the near(est) point(s) index in source point cloud for query points)
# nn: index(the near(est) point(s) index in query point cloud for source points)
# grid: constructed once for spartial discretization and point querying.
dists, idxs, nn, grid = frnn.frnn_grid_points(
        query_point, source_point, None, None, neighbor_num, max_length, grid=None, return_nn=False, return_sorted=True
)
"""