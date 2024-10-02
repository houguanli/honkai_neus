import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
import torch

#notice this aabb tree is fully built on numpy ! 
class PointCloud2AABBTree:
    def __init__(self, npz_file_path):
        self.points = self._load_point_cloud(npz_file_path)
        self.aabb_tree = None
        self.build_aabb_tree()

    def _load_point_cloud(self, npz_file_path):
        """
        Load point cloud data from a .npz file.
        Assumes the points are stored in an array called 'points'.
        """
        data = o3d.io.read_point_cloud(npz_file_path)
        points_np = np.asarray(data.points)
        return points_np
    
    def build_aabb_tree(self):
        """
        Build an AABB tree (using KDTree as an approximation), 
        and store it in self.aabb_tree.
        """
        if self.points is None:
            raise ValueError("Point cloud data is not loaded.")
        
        # Use KDTree as an approximation to the AABB tree
        self.aabb_tree = KDTree(self.points)
        print("AABB tree has been successfully built.")
    
    def query_nearest(self, query_point):
        """
        Query the nearest point to the given point.
        """
        if self.aabb_tree is None:
            raise ValueError("AABB tree is not built. Please call build_aabb_tree first.")
        
        distance, index = self.aabb_tree.query(query_point)
        return self.points[index], distance
    
    def query_nearest_batch(self, query_points):
        nearest_points, distances = [], []
        for query_point in query_points:
            point, distance = self.query_nearest(query_point)
            nearest_points.append(point)
            distances.append(distance)
        return nearest_points, distances

    def count_points_within_threshold(self, query_point, threshold=1e-3):
        """
        Count the number of points in the point cloud that are within a certain
        distance (threshold) from the given query point.
        Parameters:
        - query_point: The xyz coordinates of the query point (numpy array).
        - threshold: The distance threshold.
        Returns:
        - The number of points that are within the threshold distance.
        """
        if self.aabb_tree is None:
            raise ValueError("AABB tree is not built. Please call build_aabb_tree first.")
        
        # Query all points within the threshold distance
        indices = self.aabb_tree.query_ball_point(query_point, threshold)
        
        # Return the number of points within the threshold distance
        return len(indices)
    
    def count_points_within_threshold_batch(self, query_points, threshold=1e-3):
        """
        Returns:
        - A list with the count of points within the threshold distance for each query point.
        """
        if torch.is_tensor(query_points):
            query_points = query_points.cpu().detach().numpy()
        else:
            raise TypeError("query_points should be either a NumPy array or a PyTorch tensor.")
        counts, index = [], 0
        for query_point in query_points:
            count = self.count_points_within_threshold(query_point, threshold)
            counts.append(count)
            index = index + 1
            if index % 10000 == 0:
                print("query index ", index, " ", len(query_points))
        return counts
    
    def judge_available_points(self, query_point, threshold=1e-3, in_point_number=1):
        cnt = self.count_points_within_threshold(query_point=query_point, threshold=threshold)
        return cnt >= in_point_number
    
    def get_aabb_tree(npz_path):
        return PointCloud2AABBTree(npz_path)

class PointCloud2HashVoxel:
    def __init__(self, ply_file_path, resolution=512):
        self.point_cloud = o3d.io.read_point_cloud(ply_file_path)
        self.points = np.asarray(self.point_cloud.points)

        self.min_bound = np.min(self.points, axis=0)
        self.max_bound = np.max(self.points, axis=0)
        self.center = (self.min_bound + self.max_bound) / 2
        self.extent = (self.max_bound - self.min_bound) * 1.5
        self.grid_size = self.extent / resolution
        self.voxel_grid = self.create_voxel_grid()

    def create_voxel_grid(self):
        voxel_grid = set()
        for point in self.points:
            voxel_index = tuple(((point - self.center + self.extent / 2) // self.grid_size).astype(int))
            voxel_grid.add(voxel_index)
        return voxel_grid

    def query(self, point):
        voxel_index = tuple(((point - self.center + self.extent / 2) // self.grid_size).astype(int))
        return voxel_index in self.voxel_grid
    
    
    
# Example usage
if __name__ == "__main__":
    # Example path to a .npz file
    npz_file = "path_to_point_cloud.npz"
    
    # Instantiate the class and build the AABB tree
    point_cloud_tree = PointCloud2AABBTree(npz_file)
    point_cloud_tree.build_aabb_tree()
    # Query the nearest point
    query_point = np.array([1.0, 2.0, 3.0])  # Example query point
    nearest_point, distance = point_cloud_tree.query_nearest(query_point)
    
    print(f"The nearest point is: {nearest_point}, Distance: {distance}")
