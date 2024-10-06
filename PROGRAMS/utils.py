import logging
import pdb
from typing import Optional

import numpy as np
from logger_setup import logging
from scipy.optimize import linear_sum_assignment
from transform import FT

logger = logging.getLogger(__name__)

"""
Naming Convention:
    A_to_B represents the frame transform F_AB
    A_to_B.inv() represents the frame transform F_BA
"""


def get_identity_frame_transform():
    """Returns a numpy array representing the identity frame transform.

    Returns:
        np.ndarray: the identity frame transform.
    """
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])


def get_pcd_cost_matrix(a_pcd: np.ndarray, b_pcd: np.ndarray):
    """Computes the cost (distance) matrix between all 3D points in two pointclouds.

    Params:
        a_pcd (np.ndarray): the first pointcloud of shape (n, 3)
        b_pcd (np.ndarray): the second pointcloud of shape (m, 3)

    Returns:
        (np.ndarray): array of shape (n, m) where each value at idx i, j corresponds
            to the distance between point a_pcd[i] and b_pcd[j]
    """

    a_expanded = a_pcd[:, np.newaxis, :]  # shape becomes (3, 8, 1)
    b_expanded = b_pcd[np.newaxis, :, :]  # shape becomes (3, 1, 8)
    return np.sqrt(np.sum((a_expanded - b_expanded) ** 2, axis=2))


def compute_FT_from_point_set_pair(a: np.ndarray, b: np.ndarray):
    """Find the rotation that minimizes the mean squared distance between the
        points in the pair of sets. Assumes matrices are sorted such that each
        index in one array corresponds with the same index in the other array.

    Uses SVD shown in:
        https://cdn-uploads.piazza.com/paste/l7e4lajaoxd7hf/c11c58e8338b8842c63d0e51cde943300ad3de876a1c4c05559aa4ed0f668bff/svd1.pdf
        https://www.youtube.com/watch?v=dhzLQfDBx2Q

    Params:
        a (np.ndarray): first set of points in the pair
        b (np.ndarray): second set of points in the pair

    Returns:
        np.ndarray: 3 x 3 rotation matrix that minimizes MSE between matching
            points in the pair of point sets
    """
    if a.shape != b.shape:
        raise ValueError(
            f"Array size mismatch. a has size: {a.shape} and b has size: {b.shape}"
        )
    # the determinant is not exactly equal to 1. set a threshold for distance
    # from 1 to enforce the rotation matrix is orthogonal
    det_thresh = 0.001
    # transform pointcloud sets to local coordinates
    a_0 = a.mean(axis=0)
    a_bar = a - a_0
    b_0 = b.mean(axis=0)
    b_bar = b - b_0
    # calculate covariance matrix
    cov_mat = np.sum([np.outer(a_bar[i], b_bar[i]) for i in range(a_bar.shape[0])], axis=0)
    U, _, V_T = np.linalg.svd(cov_mat)
    V = V_T.T
    R = np.dot(V, U.T)
    det = np.linalg.det(R)
    # ensure determinant is positive
    if det < 0:
        # flip right singular matrix transpose
        V[:, 2] *= -1
        R = np.dot(V, U.T)
        det = np.linalg.det(R)
    # find the translation once the rotation is found
    t = b_0 - np.dot(R, a_0)
    # enforce the rotation matrix is orthonormal by checking the determinant is equal to 1
    if abs(det - 1.0) > det_thresh:
        raise Exception(
            f"The determinant of an orthogonal matrix must be +/-1, not {det}."
        )
    return FT(R, np.expand_dims(t, axis=0))


def pcd_to_pcd_reg(
    a_pcd: np.ndarray,
    b_pcd: np.ndarray,
    init_FT: FT,
    err_threshold: float = 5e-3,
    max_its: int = 5000,
) -> FT:
    """Takes in a pair of 3d pointclouds in numpy array format and returns
    the rotation that uses the iterative method starting on slide 16 of
    the following resource:
    https://ciis.lcsr.jhu.edu/lib/exe/fetch.php?media=courses:455-655:lectures:rigid3d3dcalculations.pdf

    Args:
        a (np.ndarray): set of vectors of shape (3, n) representing 3d pointcloud in coord frame A
        b (np.ndarray): set of vectors of shape (3, n) representing 3d pointcloud in coord frame B

    Returns:
        FT: computed frame transform A_to_B (F_AB)
    """
    assert a_pcd.shape[1] == 3, "Ensure the input vector, a, is of correct shape (3, n)"
    assert b_pcd.shape[1] == 3, "Ensure the input vector, b, is of correct shape (3, n)"
    prev_cost = np.inf
    A_to_B = init_FT
    for itr in range(max_its):
        a_pcd_in_b_frame = A_to_B.transform_pts(a_pcd)
        cost_matrix = get_pcd_cost_matrix(a_pcd_in_b_frame, b_pcd)
        # get the idxes of the points in the b_pcd that are closes to the points in a pcd
        b_idx = np.argmin(cost_matrix, axis=1)
        optimal_cost = (
            sum([cost_matrix[x, b_idx[x]] for x in range(cost_matrix.shape[0])])
            / cost_matrix.shape[0]
        )
        prev_cost = optimal_cost
        if prev_cost < err_threshold:
            break
        logger.debug(f"Itr:{itr} - Optimal Cost:{optimal_cost}")
        A_to_B = compute_FT_from_point_set_pair(a_pcd, b_pcd[b_idx, :])
    return A_to_B
