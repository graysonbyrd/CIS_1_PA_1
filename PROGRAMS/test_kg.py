import pdb

import killeengeo as kg
from transform import FT
import numpy as np
from data_processing import parse_calbody, parse_calreadings
from utils import pcd_to_pcd_reg
from scipy.spatial.transform import Rotation as R

for _ in range(100):
    rotation_degrees = [-180, -180, 0]  # Rotation around x, y, z axes in degrees
    # rotation_degrees = np.random.randint(0, 360, (1, 3)).tolist()[0]
    R_AB = R.from_euler('xyz', rotation_degrees, degrees=True).as_matrix()  # Rotation matrix
    t_AB = np.array([[1, 1, 0]])
    F_AB = FT(R_AB, t_AB)
    homogenous_FT = np.zeros((4, 4))
    homogenous_FT[:3, :3] = R_AB
    homogenous_FT[:-1, -1] = t_AB
    homogenous_FT[-1, :] = [0, 0, 0, 1]
    F_AB_kg = kg.FrameTransform(homogenous_FT)

    # b = np.array([
    #     [1, 1, 0]
    # ])

    b = np.random.rand(10, 3)

    b_in_A = F_AB.transform_pts(b)
    b_in_A_kg = F_AB_kg.transform_points(b)
    R_AB_init = R.from_euler('xyz', [-170, -170, 0], degrees=True).as_matrix()  # Rotation matrix
    F_AB_init = FT(R_AB_init, np.array([[0, 0, 0]]))
    F_AB_pred = pcd_to_pcd_reg(b_in_A, b, F_AB_init)
    # F_AB_pred_kg = kg.FrameTransform.from_points(b, b_in_A)
    pass
    # print(np.sum(F_AB.R - F_AB_kg.R))







b = np.array([
    [0, 0, 0], [1, 2, 1], [13, 14, 51], [8, 7, 6], [3, 1, 32], # [234, 41, 34], [123, 432, 12]
])
b = np.random.rand(10, 3)

# Define a known rotation and translation
rotation_degrees = [-30, 60, -60]  # Rotation around x, y, z axes in degrees
R_AB = R.from_euler('xyz', rotation_degrees, degrees=True).as_matrix()  # Rotation matrix

a = (R_AB @ b.T).T

F_D = kg.FrameTransform.from_points(a, b)

print(R_AB)
print(F_D.R)





# true_t = np.random.randint(0, 100, (1, 3))

# test_Ds = (true_rotation @ test_ds.T).T - (1, 5, -2)
# test_Ds += np.random.randn(test_ds.shape[0], test_ds.shape[1])

# test_Ds = np.array([
#     [1, 1, 1], [2, 3, 2], [14, 15, 52], [9, 8, 7]
# ])

# def main():
#     # read data
#     calbody = parse_calbody("../DATA/pa1-unknown-k-calbody.txt")
#     calreadings_frames = parse_calreadings("../DATA/pa1-unknown-k-calreadings.txt")
#     # for each frame, compute F_D between the optical tracker and EM tracker coords
#     F_D_frames = list()
#     for idx, frame in enumerate(calreadings_frames):
#         ds = calbody["d"]
#         Ds = frame["D"]
#         # if idx < 2:
#         #     continue
#         assert ds.shape == Ds.shape
#         F_D = kg.FrameTransform.from_points(test_Ds, test_ds, max_iterations=1000000)
#         # F_D = pcd_to_pcd_reg(test_ds, test_Ds)
#         print(true_rotation)
#         print(F_D.R)
#         print(true_rotation - F_D.R)
#         print(true_t - F_D.t)
#         print(test_Ds)
#         print(F_D.transform_points(test_ds))
#         # F_D_inv = F_D.inverse()
#         # print(ds - F_D.transform_points(Ds))
#         # print(Ds - F_D_inv.transform_points(ds))
#         # F_D_frames.append()
#         what = 'yes'



# if __name__ == "__main__":
#     main()
