import pdb
import open3d as o3d

import killeengeo as kg
from transform import FT
import numpy as np
from data_processing import parse_calbody, parse_calreadings
from utils import pcd_to_pcd_reg, pcd_to_pcd_with_known_correspondence
from scipy.spatial.transform import Rotation as R

def get_random_rotation_matrix():
    rotation_degrees = np.random.randint(0, 360, (1, 3)).tolist()[0]  # Rotation around x, y, z axes in degrees
    return R.from_euler('xyz', rotation_degrees, degrees=True).as_matrix()  # Rotation matrix

def get_transform_with_random_rotation_zero_vector_translation():
    R = get_random_rotation_matrix()
    t = np.zeros((1, 3))
    return FT(R, t)

def visualize_numpy_arrays_as_pcds(arrays):
    pcds = list()
    for arr in arrays:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(arr)
        pcds.append(pcd)
    o3d.visualization.draw_geometries(pcds)

def get_best_fit_after_n_random_inits(ds, Ds, n_random_inits):
    best_FT = None
    best_diff = 10000000
    for i in range(n_random_inits):
        init_FT = get_transform_with_random_rotation_zero_vector_translation()
        F_D_random_init = pcd_to_pcd_reg(Ds, ds, init_FT)
        Ds_in_ds = F_D_random_init.transform_pts(Ds)
        # calculate diff
        diff = np.sqrt(np.sum((Ds_in_ds - ds)**2))
        diff_vanilla = np.sum(Ds_in_ds - ds)
        print(Ds_in_ds.astype('int'))
        print(ds)
        print(f"{i}: {diff}")
        print(f"{i}: {diff_vanilla}")
        if diff < best_diff:
            best_diff = diff
            best_FT = F_D_random_init
    return best_FT

def test_consistency_kg_vs_custom():
    # read data
    calbody = parse_calbody("../DATA/pa1-unknown-k-calbody.txt")
    calreadings_frames = parse_calreadings("../DATA/pa1-unknown-k-calreadings.txt")
    # for each frame, compute F_D between the optical tracker and EM tracker coords
    F_D_frames = list()
    print(f"Checking consistency...")
    for idx, frame in enumerate(calreadings_frames):
        ds = calbody["d"]
        Ds = frame["D"]
        assert ds.shape == Ds.shape
        F_D_kg = kg.FrameTransform.from_points(ds, Ds, max_iterations=1000000)
        F_D = pcd_to_pcd_reg(Ds, ds)
        print(f"Diff: {np.sqrt(np.sum((F_D_kg.R - F_D.R)**2))}")
    print(f"Consistency check done...\n\n\n")

def test_performance_F_D():
        # read data
    calbody = parse_calbody("../DATA/pa1-unknown-k-calbody.txt")
    calreadings_frames = parse_calreadings("../DATA/pa1-unknown-k-calreadings.txt")
    # for each frame, compute F_D between the optical tracker and EM tracker coords
    F_D_frames = list()
    print(f"Checking consistency...")
    for idx, frame in enumerate(calreadings_frames):
        ds = calbody["d"]
        Ds = frame["D"]
        assert ds.shape == Ds.shape
        F_D_kg = kg.FrameTransform.from_points(ds, Ds, max_iterations=1000000)
        Ds_in_ds_kg = F_D_kg.transform_points(Ds)
        F_D = pcd_to_pcd_reg(Ds, ds)
        Ds_in_ds_custom = F_D.transform_pts(Ds)
        print(f"Diff kg: {np.sqrt(np.sum((Ds_in_ds_kg - ds)**2))}")
        print(f"Diff custom: {np.sqrt(np.sum((Ds_in_ds_custom - ds)**2))}")
        temp = np.zeros_like(ds)
        temp += Ds_in_ds_custom
        visualize_numpy_arrays_as_pcds([ds, temp])
        # pcd_1 = o3d.geometry.PointCloud()
        # pcd_2 = o3d.geometry.PointCloud()
        # pcd_1.points = o3d.utility.Vector3dVector(ds)
        # pcd_2.points = o3d.utility.Vector3dVector(Ds_in_ds_custom)
        # o3d.visualization.draw_geometries([pcd_1, pcd_2])
        # visualize_numpy_arrays_as_pcds([Ds_in_ds_custom, ds])
    print(f"Consistency check done...\n\n\n")

def test_performance_F_A():
        # read data
    calbody = parse_calbody("../DATA/pa1-unknown-k-calbody.txt")
    calreadings_frames = parse_calreadings("../DATA/pa1-unknown-k-calreadings.txt")
    # for each frame, compute F_D between the optical tracker and EM tracker coords
    F_D_frames = list()
    print(f"Checking consistency...")
    for idx, frame in enumerate(calreadings_frames):
        ds = calbody["a"]
        Ds = frame["A"]
        assert ds.shape == Ds.shape
        F_D_kg = kg.FrameTransform.from_points(ds, Ds, max_iterations=1000000)
        Ds_in_ds_kg = F_D_kg.transform_points(Ds)
        F_D = pcd_to_pcd_reg(Ds, ds)
        Ds_in_ds_custom = F_D.transform_pts(Ds)
        print(f"Diff kg: {np.sqrt(np.sum((Ds_in_ds_kg - ds)**2))}")
        print(f"Diff custom: {np.sqrt(np.sum((Ds_in_ds_custom - ds)**2))}")
        temp = np.zeros_like(ds)
        temp += Ds_in_ds_custom
        visualize_numpy_arrays_as_pcds([ds, temp])
        # if idx == 3:
        #     pcd_1 = o3d.geometry.PointCloud()
        #     pcd_2 = o3d.geometry.PointCloud()
        #     pcd_1.points = o3d.utility.Vector3dVector(ds)
        #     pcd_2.points = o3d.utility.Vector3dVector(Ds)
        #     o3d.visualization.draw_geometries([pcd_1, pcd_2])
        # print(f"Diff KG: {np.sqrt(np.sum((F_D_kg.R - F_D.R)**2))}")
    print(f"Consistency check done...\n\n\n")

def test_performance_F_A_w_random_init():
    # read data
    calbody = parse_calbody("../DATA/pa1-unknown-k-calbody.txt")
    calreadings_frames = parse_calreadings("../DATA/pa1-unknown-k-calreadings.txt")
    # for each frame, compute F_D between the optical tracker and EM tracker coords
    F_D_frames = list()
    print(f"Checking consistency...")
    for idx, frame in enumerate(calreadings_frames):
        ds = calbody["a"]
        Ds = frame["A"]
        assert ds.shape == Ds.shape
        F_D_kg = kg.FrameTransform.from_points(ds, Ds, max_iterations=1000000)
        Ds_in_ds_kg = F_D_kg.transform_points(Ds)
        F_D = get_best_fit_after_n_random_inits(ds, Ds, 30)
        Ds_in_ds_custom = F_D.transform_pts(Ds)
        print(f"Diff kg: {np.sqrt(np.sum((Ds_in_ds_kg - ds)**2))}")
        print(f"Diff custom: {np.sqrt(np.sum((Ds_in_ds_custom - ds)**2))}")
        temp = np.zeros_like(ds)
        temp += Ds_in_ds_custom
        visualize_numpy_arrays_as_pcds([ds, temp])
        # if idx == 3:
        #     pcd_1 = o3d.geometry.PointCloud()
        #     pcd_2 = o3d.geometry.PointCloud()
        #     pcd_1.points = o3d.utility.Vector3dVector(ds)
        #     pcd_2.points = o3d.utility.Vector3dVector(Ds)
        #     o3d.visualization.draw_geometries([pcd_1, pcd_2])
        # print(f"Diff KG: {np.sqrt(np.sum((F_D_kg.R - F_D.R)**2))}")
    print(f"Consistency check done...\n\n\n")

def check_if_random_init_changes_results():
    # read data
    calbody = parse_calbody("../DATA/pa1-unknown-k-calbody.txt")
    calreadings_frames = parse_calreadings("../DATA/pa1-unknown-k-calreadings.txt")
    # for each frame, compute F_D between the optical tracker and EM tracker coords
    F_D_frames = list()
    print(f"Checking if random init changes results...")
    for idx, frame in enumerate(calreadings_frames):
        ds = calbody["d"]
        Ds = frame["D"]
        assert ds.shape == Ds.shape
        init_T = get_transform_with_random_rotation_zero_vector_translation()
        F_D_random_init = pcd_to_pcd_reg(Ds, ds, init_T)
        F_D_identity_init = pcd_to_pcd_reg(Ds, ds)
        Ds_in_ds = F_D_identity_init.transform_pts(Ds)
        print(f"Diff: {np.sum(Ds_in_ds - ds)}")
        print(f"Diff: {np.sum(F_D_random_init.R - F_D_identity_init.R)}")
    print(f"Consistency check done...\n\n\n")

def check_if_random_init_improves_results():
    # read data
    calbody = parse_calbody("../DATA/pa1-unknown-i-calbody.txt")
    calreadings_frames = parse_calreadings("../DATA/pa1-unknown-i-calreadings.txt")
    # for each frame, compute F_D between the optical tracker and EM tracker coords
    F_D_frames = list()
    print(f"Checking if random init improves results...")
    total_diff_identity = 0
    total_diff_best_random_init = 0
    for idx, frame in enumerate(calreadings_frames):
        ds = calbody["d"]
        Ds = frame["D"]
        assert ds.shape == Ds.shape
        best_F_D_random_init = get_best_fit_after_n_random_inits(ds, Ds, 30)
        F_D_identity_init = pcd_to_pcd_reg(Ds, ds)
        Ds_in_ds_identity = F_D_identity_init.transform_pts(Ds)
        Ds_in_ds_best_random_init = best_F_D_random_init.transform_pts(Ds)
        diff_identity = np.sqrt(np.sum((Ds_in_ds_identity - ds)**2))
        diff_best_random_init = np.sqrt(np.sum((Ds_in_ds_best_random_init - ds)**2))
        print(f"\n\nDiff Identity: {diff_identity}")
        print(f"Diff Best Random Init: {diff_best_random_init}")
        total_diff_identity += diff_identity
        total_diff_best_random_init += diff_best_random_init
    print(f"Average diff identity: {total_diff_identity / idx}")
    print(f"Average diff best random init: {total_diff_best_random_init / idx}")
    print(f"Improve results check done...\n\n\n")

def check_if_aligning_centroid_changes_results():
    """We already align the centroids in the algorithm."""
    pass
    # # read data
    # calbody = parse_calbody("../DATA/pa1-unknown-k-calbody.txt")
    # calreadings_frames = parse_calreadings("../DATA/pa1-unknown-k-calreadings.txt")
    # # for each frame, compute F_D between the optical tracker and EM tracker coords
    # F_D_frames = list()
    # print(f"Checking consistency...")
    # for idx, frame in enumerate(calreadings_frames):
    #     ds = calbody["d"]
    #     Ds = frame["D"]
    #     assert ds.shape == Ds.shape
    #     F_D_kg = kg.FrameTransform.from_points(ds, Ds, max_iterations=1000000)
    #     init_T = get_transform_with_random_rotation_zero_vector_translation()
    #     F_D = pcd_to_pcd_reg(Ds, ds)
    #     print(f"Diff: {np.sum(F_D_kg.R - F_D.R)}")
    # print(f"Consistency check done...\n\n\n")

def get_best_from_n_random_inits_and_compare_to_identity_init():
    # read data
    calbody = parse_calbody("../DATA/pa1-unknown-k-calbody.txt")
    calreadings_frames = parse_calreadings("../DATA/pa1-unknown-k-calreadings.txt")
    # for each frame, compute F_D between the optical tracker and EM tracker coords
    F_D_frames = list()
    print(f"Checking if random init changes results...")
    for idx, frame in enumerate(calreadings_frames):
        ds = calbody["d"]
        Ds = frame["D"]
        assert ds.shape == Ds.shape
        init_T = get_transform_with_random_rotation_zero_vector_translation()
        F_D_random_init = pcd_to_pcd_reg(Ds, ds, init_T)
        F_D_identity_init = pcd_to_pcd_reg(Ds, ds)
        print(f"Diff: {np.sum(F_D_random_init.R - F_D_identity_init.R)}")
    print(f"Consistency check done...\n\n\n")

def visualize_pa_data_pointclouds_D_in_open3d():
    # read data
    calbody = parse_calbody("../DATA/pa1-debug-a-calbody.txt")
    calreadings_frames = parse_calreadings("../DATA/pa1-debug-a-calreadings.txt")
    # for each frame, compute F_D between the optical tracker and EM tracker coords
    F_D_frames = list()
    print(f"Checking if random init changes results...")
    for idx, frame in enumerate(calreadings_frames):
        ds = calbody["d"]
        Ds = frame["D"]
        assert ds.shape == Ds.shape
        visualize_numpy_arrays_as_pcds(ds, Ds)
        pcd_1 = o3d.geometry.PointCloud()
        pcd_2 = o3d.geometry.PointCloud()
        pcd_1.points = o3d.utility.Vector3dVector(ds)
        pcd_2.points = o3d.utility.Vector3dVector(Ds)
        # o3d.visualization.draw_geometries([pcd_2])

        o3d.visualization.draw_geometries([pcd_1, pcd_2])
    print(f"Consistency check done...\n\n\n")

def visualize_pa_data_pointclouds_A_in_open3d():
    # read data
    calbody = parse_calbody("../DATA/pa1-unknown-k-calbody.txt")
    calreadings_frames = parse_calreadings("../DATA/pa1-unknown-k-calreadings.txt")
    # for each frame, compute F_D between the optical tracker and EM tracker coords
    F_D_frames = list()
    print(f"Checking if random init changes results...")
    for idx, frame in enumerate(calreadings_frames):
        ds = calbody["a"]
        Ds = frame["A"]
        print(ds)
        print(Ds)
        assert ds.shape == Ds.shape
        visualize_numpy_arrays_as_pcds([ds, Ds])
        pcd_1 = o3d.geometry.PointCloud()
        pcd_2 = o3d.geometry.PointCloud()
        pcd_1.points = o3d.utility.Vector3dVector(ds)
        pcd_2.points = o3d.utility.Vector3dVector(Ds)
        # o3d.visualization.draw_geometries([pcd_2])

        o3d.visualization.draw_geometries([pcd_1, pcd_2])
    print(f"Consistency check done...\n\n\n")

def print_pa_data_pointclouds_A_in_open3d():
    # read data
    calbody = parse_calbody("../DATA/pa1-debug-a-calbody.txt")
    calreadings_frames = parse_calreadings("../DATA/pa1-debug-a-calreadings.txt")
    # for each frame, compute F_D between the optical tracker and EM tracker coords
    F_D_frames = list()
    print(f"Checking if random init changes results...")
    for idx, frame in enumerate(calreadings_frames):
        ds = calbody["a"]
        Ds = frame["A"]
        print(ds)
        print(Ds)
        assert ds.shape == Ds.shape

    print(f"Consistency check done...\n\n\n")

def sanity_check_icp_algorithm():
    # get random target pointcloud
    target = np.random.rand(30, 3)
    # get rotation vector
    rotation_degrees = [30, 45, 0]  # Rotation around x, y, z axes in degrees
    R_AB = R.from_euler('xyz', rotation_degrees, degrees=True).as_matrix()  # Rotation matrix
    t_AB = np.array([[1, 1, 0]])
    source = (R_AB @ target.T).T + t_AB
    init_FT = FT(R_AB, np.zeros((1, 3)))
    # get the predicted rotation vector
    F_AB_pred_kg = kg.FrameTransform.from_points(source, target)
    F_AB_pred_custom = pcd_to_pcd_reg(target, source)
    F_AB_pred_custom_good_init = pcd_to_pcd_reg(target, source, init_FT)
    print(R_AB)
    print(F_AB_pred_kg.R)
    print(F_AB_pred_custom.R)
    print(F_AB_pred_custom_good_init.R)
    what = 'yes'

def test_performance_F_D_w_known_correspondence():
    # read data
    calbody = parse_calbody("../DATA/pa1-unknown-k-calbody.txt")
    calreadings_frames = parse_calreadings("../DATA/pa1-unknown-k-calreadings.txt")
    # for each frame, compute F_D between the optical tracker and EM tracker coords
    F_D_frames = list()
    print(f"Checking consistency...")
    for idx, frame in enumerate(calreadings_frames):
        ds = calbody["d"]
        Ds = frame["D"]
        assert ds.shape == Ds.shape
        F_D_kg = kg.FrameTransform.from_points(ds, Ds, max_iterations=1000000)
        Ds_in_ds_kg = F_D_kg.transform_points(Ds)
        F_D = pcd_to_pcd_with_known_correspondence(Ds, ds)
        # F_D = pcd_to_pcd_reg(Ds, ds)
        Ds_in_ds_custom = F_D.transform_pts(Ds)
        print(f"Diff kg: {np.sqrt(np.sum((Ds_in_ds_kg - ds)**2))}")
        print(f"Diff custom: {np.sqrt(np.sum((Ds_in_ds_custom - ds)**2))}")
        temp = np.zeros_like(ds)
        temp += Ds_in_ds_custom
        # visualize_numpy_arrays_as_pcds([ds, temp])
        # pcd_1 = o3d.geometry.PointCloud()
        # pcd_2 = o3d.geometry.PointCloud()
        # pcd_1.points = o3d.utility.Vector3dVector(ds)
        # pcd_2.points = o3d.utility.Vector3dVector(Ds_in_ds_custom)
        # o3d.visualization.draw_geometries([pcd_1, pcd_2])
        # visualize_numpy_arrays_as_pcds([Ds_in_ds_custom, ds])
    print(f"Consistency check done...\n\n\n")

def test_performance_F_A_w_known_correspondence():
    # read data
    calbody = parse_calbody("../DATA/pa1-unknown-k-calbody.txt")
    calreadings_frames = parse_calreadings("../DATA/pa1-unknown-k-calreadings.txt")
    # for each frame, compute F_D between the optical tracker and EM tracker coords
    F_D_frames = list()
    print(f"Checking consistency...")
    for idx, frame in enumerate(calreadings_frames):
        ds = calbody["a"]
        Ds = frame["A"]
        assert ds.shape == Ds.shape
        F_D_kg = kg.FrameTransform.from_points(ds, Ds, max_iterations=1000000)
        Ds_in_ds_kg = F_D_kg.transform_points(Ds)
        F_D = pcd_to_pcd_with_known_correspondence(Ds, ds)
        # F_D = pcd_to_pcd_reg(Ds, ds)
        Ds_in_ds_custom = F_D.transform_pts(Ds)
        print(f"Diff kg: {np.sqrt(np.sum((Ds_in_ds_kg - ds)**2))}")
        print(f"Diff custom: {np.sqrt(np.sum((Ds_in_ds_custom - ds)**2))}")
        temp = np.zeros_like(ds)
        temp += Ds_in_ds_custom
        # visualize_numpy_arrays_as_pcds([ds, temp])
        # pcd_1 = o3d.geometry.PointCloud()
        # pcd_2 = o3d.geometry.PointCloud()
        # pcd_1.points = o3d.utility.Vector3dVector(ds)
        # pcd_2.points = o3d.utility.Vector3dVector(Ds_in_ds_custom)
        # o3d.visualization.draw_geometries([pcd_1, pcd_2])
        # visualize_numpy_arrays_as_pcds([Ds_in_ds_custom, ds])
    print(f"Consistency check done...\n\n\n")

if __name__ == '__main__':
    test_performance_F_A_w_known_correspondence()
    # test_consistency_kg_vs_custom()
    # check_if_random_init_changes_results()
    # check_if_random_init_improves_results()
    # visualize_pa_data_pointclouds_A_in_open3d()
    # sanity_check_icp_algorithm()
    # print_pa_data_pointclouds_A_in_open3d()
    # test_performance_F_A_w_random_init()

# for _ in range(100):
#     rotation_degrees = [-180, -180, 0]  # Rotation around x, y, z axes in degrees
#     # rotation_degrees = np.random.randint(0, 360, (1, 3)).tolist()[0]
#     R_AB = R.from_euler('xyz', rotation_degrees, degrees=True).as_matrix()  # Rotation matrix
#     t_AB = np.array([[1, 1, 0]])
#     F_AB = FT(R_AB, t_AB)
#     homogenous_FT = np.zeros((4, 4))
#     homogenous_FT[:3, :3] = R_AB
#     homogenous_FT[:-1, -1] = t_AB
#     homogenous_FT[-1, :] = [0, 0, 0, 1]
#     F_AB_kg = kg.FrameTransform(homogenous_FT)

#     # b = np.array([
#     #     [1, 1, 0]
#     # ])

#     b = np.random.rand(10, 3)

#     b_in_A = F_AB.transform_pts(b)
#     b_in_A_kg = F_AB_kg.transform_points(b)
#     R_AB_init = R.from_euler('xyz', [-170, -170, 0], degrees=True).as_matrix()  # Rotation matrix
#     F_AB_init = FT(R_AB_init, np.array([[0, 0, 0]]))
#     F_AB_pred = pcd_to_pcd_reg(b_in_A, b, F_AB_init)
#     # F_AB_pred_kg = kg.FrameTransform.from_points(b, b_in_A)
#     pass
#     # print(np.sum(F_AB.R - F_AB_kg.R))







# b = np.array([
#     [0, 0, 0], [1, 2, 1], [13, 14, 51], [8, 7, 6], [3, 1, 32], # [234, 41, 34], [123, 432, 12]
# ])
# b = np.random.rand(10, 3)

# # Define a known rotation and translation
# rotation_degrees = [-30, 60, -60]  # Rotation around x, y, z axes in degrees
# R_AB = R.from_euler('xyz', rotation_degrees, degrees=True).as_matrix()  # Rotation matrix

# a = (R_AB @ b.T).T

# F_D = kg.FrameTransform.from_points(a, b)

# print(R_AB)
# print(F_D.R)