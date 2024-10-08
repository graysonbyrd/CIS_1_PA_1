import logging

import numpy as np
from logger_setup import logging
from transform import FT

logger = logging.getLogger(__name__)
"""
Naming Convention:
    A_to_B represents the frame transform F_AB
    A_to_B.inv() represents the frame transform F_BA
"""


def pcd_to_pcd_reg_w_known_correspondence(a: np.ndarray, b: np.ndarray):
    """Finds the optimal transform that aligns point cloud A to point cloud B
    using linear least squares via Singular Value Decomposition (SVD).

    Uses SVD shown in:
        https://cdn-uploads.piazza.com/paste/l7e4lajaoxd7hf/c11c58e8338b8842c63d0e51cde943300ad3de876a1c4c05559aa4ed0f668bff/svd1.pdf
        https://www.youtube.com/watch?v=dhzLQfDBx2Q

    Args:
        a (np.array): the target pointcloud
        b (np.array): the source pointcloud

    Returns:
        FT: Frame transform such that the mean squared error between FT(a)
            and b is minimized.
    """
    # ompute the centroids of both point clouds
    centroid_A = np.mean(a, axis=0)
    centroid_B = np.mean(b, axis=0)
    # align the points in local coordinate frame
    A_centered = a - centroid_A
    B_centered = b - centroid_B
    # compute the covariance matrix
    H = np.dot(A_centered.T, B_centered)
    # Singular Value Decomposition (SVD)
    U, _, Vt = np.linalg.svd(H)
    # compute the optimal rotation matrix
    R = np.dot(Vt.T, U.T)
    # handle reflection case (ensure a proper rotation)
    # explanation shown in part VI of: https://ieeexplore.ieee.org/document/4767965
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    # compute translation vector
    t = np.expand_dims(centroid_B - (R @ centroid_A.T).T, axis=0)
    return FT(R, t)
