import pdb

import killeengeo as kg
import numpy as np
from data_processing import parse_calbody, parse_calreadings
from utils import pcd_to_pcd_reg
from scipy.spatial.transform import Rotation as R


test_ds = np.array([
    [0, 0, 0], [1, 2, 1], [13, 14, 51], [8, 7, 6], [3, 1, 32], # [234, 41, 34], [123, 432, 12]
])
test_ds = np.random.rand(10, 3)

# Define a known rotation and translation
rotation_degrees = [-30, 60, -60]  # Rotation around x, y, z axes in degrees
true_rotation = R.from_euler('xyz', rotation_degrees, degrees=True).as_matrix()  # Rotation matrix
true_t = np.random.randint(0, 100, (1, 3))

test_Ds = (true_rotation @ test_ds.T).T - (1, 5, -2)
test_Ds += np.random.randn(test_ds.shape[0], test_ds.shape[1])

# test_Ds = np.array([
#     [1, 1, 1], [2, 3, 2], [14, 15, 52], [9, 8, 7]
# ])

def main():
    # read data
    calbody = parse_calbody("../DATA/pa1-unknown-k-calbody.txt")
    calreadings_frames = parse_calreadings("../DATA/pa1-unknown-k-calreadings.txt")
    # for each frame, compute F_D between the optical tracker and EM tracker coords
    F_D_frames = list()
    for idx, frame in enumerate(calreadings_frames):
        ds = calbody["d"]
        Ds = frame["D"]
        # if idx < 2:
        #     continue
        assert ds.shape == Ds.shape
        F_D = kg.FrameTransform.from_points(test_Ds, test_ds, max_iterations=1000000)
        # F_D = pcd_to_pcd_reg(test_ds, test_Ds)
        print(true_rotation)
        print(F_D.R)
        print(true_rotation - F_D.R)
        print(true_t - F_D.t)
        print(test_Ds)
        print(F_D.transform_points(test_ds))
        # F_D_inv = F_D.inverse()
        # print(ds - F_D.transform_points(Ds))
        # print(Ds - F_D_inv.transform_points(ds))
        # F_D_frames.append()
        what = 'yes'



if __name__ == "__main__":
    main()
