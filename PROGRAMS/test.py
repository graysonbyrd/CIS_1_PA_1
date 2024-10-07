import open3d as o3d
import numpy as np

# Create some random points for the point cloud
num_points = 1000
points = np.random.rand(num_points, 3)

# Create an Open3D PointCloud object
pcd = o3d.geometry.PointCloud()

# Assign the generated points to the PointCloud object
pcd.points = o3d.utility.Vector3dVector(points)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd], window_name="Basic Point Cloud")