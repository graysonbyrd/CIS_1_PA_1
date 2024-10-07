import numpy as np

def find_optimal_rotation(A, B):
    """
    Finds the optimal rotation matrix that aligns point cloud A to point cloud B.
    A and B are Nx3 matrices of corresponding 3D points.
    """
    # Step 1: Compute the centroids of both point clouds
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # Step 2: Center the points by subtracting the centroids
    A_centered = A - centroid_A
    B_centered = B - centroid_B
    
    # Step 3: Compute the covariance matrix
    H = np.dot(A_centered.T, B_centered)
    
    # Step 4: Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(H)
    
    # Step 5: Compute the optimal rotation matrix
    R = np.dot(Vt.T, U.T)
    
    # Handle reflection case (ensure a proper rotation)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    
    return R

import numpy as np

def apply_rotation(points, R):
    """
    Applies a rotation matrix R to a set of points.
    points is an Nx3 matrix of 3D points.
    R is a 3x3 rotation matrix.
    """
    return np.dot(points, R.T)

def check_algorithm_with_perfect_match():
    """
    Tests the rotation alignment algorithm on two point clouds that perfectly match.
    One point cloud is a rotated version of the other.
    """
    # Generate a random set of 3D points (Nx3 matrix)
    np.random.seed(42)  # For reproducibility
    A = np.random.rand(10, 3)  # Generate 10 random 3D points
    
    # Define a known rotation matrix (for example, a 45-degree rotation around the Z-axis)
    theta = np.pi / 4  # 45 degrees
    R_known = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])
    
    # Apply the known rotation to point cloud A to create point cloud B
    B = apply_rotation(A, R_known)
    
    # Use the previously defined function to find the optimal rotation matrix
    R_computed = find_optimal_rotation(A, B)
    
    print("Known rotation matrix:")
    print(R_known)
    
    print("\nComputed rotation matrix:")
    print(R_computed)
    
    # Apply the computed rotation to point cloud A to verify alignment
    A_rotated = apply_rotation(A, R_computed)
    
    # Compute the mean squared error (should be very small for a perfect match)
    mse = np.mean(np.linalg.norm(A_rotated - B, axis=1)**2)
    
    print("\nMean Squared Error (MSE):", mse)
    
    # Check if the computed rotation matrix is close to the known rotation matrix
    if np.allclose(R_known, R_computed):
        print("\nThe computed rotation matrix is correct.")
    else:
        print("\nThe computed rotation matrix is incorrect.")

# Run the test function
check_algorithm_with_perfect_match()