import numpy as np
from scipy.spatial import KDTree

def icp(source, target, max_iterations=100, tolerance=1e-6):
    """
    Perform ICP pointcloud registration using SVD.
    
    :param source: Nx3 numpy array of the source point cloud.
    :param target: Nx3 numpy array of the target point cloud.
    :param max_iterations: Maximum number of iterations to run the ICP algorithm.
    :param tolerance: Threshold for convergence.
    
    :return: Transformation matrix (4x4), transformed source cloud.
    """
    
    def compute_centroid(points):
        """ Compute the centroid of a set of points. """
        return np.mean(points, axis=0)
    
    def compute_transformation(src, tgt):
        """ Compute rotation (R) and translation (t) using SVD. """
        # Compute centroids of both source and target
        centroid_src = compute_centroid(src)
        centroid_tgt = compute_centroid(tgt)
        
        # Center the points
        src_centered = src - centroid_src
        tgt_centered = tgt - centroid_tgt
        
        # Compute covariance matrix
        H = np.dot(src_centered.T, tgt_centered)
        
        # Compute SVD
        U, _, Vt = np.linalg.svd(H)
        
        # Compute rotation
        R = np.dot(Vt.T, U.T)
        
        # Handle special case of reflection
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)
        
        # Compute translation
        t = centroid_tgt - np.dot(R, centroid_src)
        
        return R, t

    def apply_transformation(points, R, t):
        """ Apply transformation to the point cloud. """
        return np.dot(points, R.T) + t
    
    # Initialize transformation as identity matrix
    prev_error = np.inf
    transformation_matrix = np.eye(4)
    
    for i in range(max_iterations):
        # Step 1: Find nearest neighbors in the target point cloud
        tree = KDTree(target)
        distances, indices = tree.query(source)
        corresponding_points = target[indices]
        
        # Step 2: Compute the best transformation between the current source and corresponding points
        R, t = compute_transformation(source, corresponding_points)
        
        # Apply transformation to the source point cloud
        source = apply_transformation(source, R, t)
        
        # Update the transformation matrix
        current_transformation = np.eye(4)
        current_transformation[:3, :3] = R
        current_transformation[:3, 3] = t
        transformation_matrix = np.dot(current_transformation, transformation_matrix)
        
        # Step 3: Check for convergence (based on mean square error)
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            print(f"Converged after {i+1} iterations.")
            break
        prev_error = mean_error
    
    return transformation_matrix, source

# Example usage
if __name__ == "__main__":
    # Generate random point clouds as an example
    np.random.seed(42)
    
    # Target point cloud (Q)
    target_cloud = np.random.rand(100, 3)
    
    # Apply a known transformation to create the source point cloud (P)
    true_rotation = np.array([[0.866, -0.5, 0],
                              [0.5, 0.866, 0],
                              [0, 0, 1]])
    true_translation = np.array([0.5, 0.3, 0.2])
    source_cloud = np.dot(target_cloud, true_rotation.T) + true_translation
    # Perform ICP
    transformation, aligned_source = icp(source_cloud, target_cloud)
    print("Estimated Transformation Matrix:\n", transformation)