import models.py_fricp as py_fricp
import numpy as np  
import open3d as o3d

# Create a new instance of the Fricp class, for now only the double precision version is available
fricp = py_fricp.PY_FRICPd()

'''
Method 1: Set the source and target point clouds using numpy arrays
Make sure the numpy array is of shape (3, N) where N is the number of points
Use the .T attribute to transpose the array if needed.
'''
# Load the source and target point clouds
source = o3d.io.read_point_cloud("data/source.ply")
target = o3d.io.read_point_cloud("data/target.ply")

# convert the point clouds to numpy arrays
# As the the point clouds read by the open3d library are already float64, we don't need to convert them,
# but if the point clouds are of any other type, convert them to float64
source = np.asarray(source.points)#.astype(np.float64)
target = np.asarray(target.points)#.astype(np.float64)
print("Source type:", source.dtype)
print("Target type:", target.dtype)
# reshape as 3 x N
source = source.T
target = target.T
# Set the source and target points
fricp.set_points(source_point=source, target_point=target)

'''
Method 2: Give the path to the source and target point clouds
The point clouds can be and only can be in .ply or .obj format
'''
# fricp.set_points_from_file("data/source.ply", "data/target.ply")

# result is a numpy array of shape (4, 4)
result = fricp.run_icp(method=3)
print("Transformation matrix shape: ", result.shape)
print("Transformation matrix: \n", result)