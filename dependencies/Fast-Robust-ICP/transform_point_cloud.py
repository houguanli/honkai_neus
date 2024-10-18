import open3d as o3d
import numpy as np

def load_ply(filename):
    """Load the point cloud from a .ply file."""
    point_cloud = o3d.io.read_point_cloud(filename)
    return point_cloud

def apply_transformations(point_cloud, rotation_matrix, translation_vector):
    """
    Apply rotation and translation to the point cloud.

    :param point_cloud: The original point cloud data.
    :param rotation_matrix: 3x3 numpy array representing rotation.
    :param translation_vector: 1x3 numpy array representing translation.
    """
    # Convert the point cloud to numpy array for manipulation
    points = np.asarray(point_cloud.points)
    
    # Apply rotation
    rotated_points = np.dot(points, rotation_matrix.T)
    
    # Apply translation
    transformed_points = rotated_points + translation_vector
    import pdb; pdb.set_trace()
    # Update the point cloud with the transformed points
    point_cloud.points = o3d.utility.Vector3dVector(transformed_points)

    return point_cloud

def save_ply(point_cloud, output_filename):
    """Save the transformed point cloud to a .ply file."""
    o3d.io.write_point_cloud(output_filename, point_cloud, write_ascii=True)

def main(input_file, output_file, rotation_degrees, translation):
    # Load the original point cloud
    point_cloud = load_ply(input_file)
    
    # Convert degrees to radians for rotation
    rotation_radians = np.radians(rotation_degrees)
    
    # Create rotation matrix for x, y, z rotations
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rotation_radians[0]), -np.sin(rotation_radians[0])],
                   [0, np.sin(rotation_radians[0]), np.cos(rotation_radians[0])]])

    Ry = np.array([[np.cos(rotation_radians[1]), 0, np.sin(rotation_radians[1])],
                   [0, 1, 0],
                   [-np.sin(rotation_radians[1]), 0, np.cos(rotation_radians[1])]])

    Rz = np.array([[np.cos(rotation_radians[2]), -np.sin(rotation_radians[2]), 0],
                   [np.sin(rotation_radians[2]), np.cos(rotation_radians[2]), 0],
                   [0, 0, 1]])
    
    # Final rotation matrix is Rz * Ry * Rx (ZYX order)
    rotation_matrix = Rz @ Ry @ Rx

    # Apply rotation and translation to the point cloud
    transformed_cloud = apply_transformations(point_cloud, rotation_matrix, translation)
    
    # Save the transformed point cloud
    save_ply(transformed_cloud, output_file)

if __name__ == "__main__":
    # User input parameters
    input_file = "data/source.ply"  # Replace with the path to your input .ply file
    output_file = "bunny_transform.ply"  # Replace with the desired output path

    # Define rotation angles (in degrees) for x, y, z axes
    rotation_degrees = [45, 30, 60]  # Example: rotate 45 degrees on x, 30 on y, and 60 on z

    # Define translation vector [tx, ty, tz]
    translation = np.array([0.5, 0.5, 1.0])  # Example translation

    # Execute the transformation and save
    main(input_file, output_file, rotation_degrees, translation)