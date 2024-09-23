import numpy as np
from scipy.spatial import KDTree
import open3d as o3d

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
    
    
    def judge_available_points(self, query_point, threshold=1e-3, in_point_number=1):
        cnt = self.count_points_within_threshold(query_point=query_point, threshold=threshold)
        return cnt >= in_point_number
    
    def get_aabb_tree(npz_path):
        return PointCloud2AABBTree(npz_path)

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
