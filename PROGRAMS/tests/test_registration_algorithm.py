import pdb

import killeengeo as kg
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils import pcd_to_pcd_reg


def test_pcd_to_pcd_registration():
    errors = list()
    for _ in range(50):
        # generate a random pointcloud set
        pcd_size = 10
        pcd_1 = np.random.randint(0, 1001, (pcd_size, 3))
        # get a random rotation matrix and translation matrix
        R_deg = np.random.randint(0, 360, (1, 3))  # Rot in degrees
        true_R = R.from_euler("xyz", R_deg, degrees=True).as_matrix().squeeze()
        true_t = np.random.randint(0, 100, (1, 3))
        # create a second pointcloud using the true_R and true_t
        pcd_2: np.ndarray = (true_R @ pcd_1.T).T + true_t
        # add random noise to pcd_2 to simulate real world scenario
        r, c = pcd_2.shape
        # pcd_2 += np.random.randn(r, c)
        # get best fit transform
        # pdb.set_trace()
        # ft = pcd_to_pcd_reg(pcd_2, pcd_1)
        ft = kg.FrameTransform.from_points(pcd_1, pcd_2)
        # get the mse associated with the frame transform
        # estimated_pcd_1 = ft.transform_pts(pcd_2)
        estimated_pcd_1 = ft.transform_points(pcd_2)
        mse = np.sqrt(np.sum((estimated_pcd_1 - pcd_1) ** 2, axis=0))
        errors.append(mse)
    max_error = max(errors)
    assert max_error < 1


