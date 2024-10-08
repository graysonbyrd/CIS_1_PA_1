import numpy as np
from data_processing import parse_calbody, parse_calreadings
from transform import FT
from utils import pcd_to_pcd_reg_w_known_correspondence


def compute_C_i_expected(calbody_file_path: str, calreadings_file_path: str):
    """Takes in the calibration dataset prefix (e.g. "pa1-debug-c-").
    Loads the relevant data, and follows the procedures outlined
    in Question 4 under Assignment 1 in CIS I PA 1 to compute the C_i_expected
    for each frame in the calibration dataset.

    Params:
        calibration_dataset_prefix (str): the prefix of the calibration
            dataset

    Returns:
        np.ndarray: the computed C_i_expected for each frame in the
            calibration dataset.
    """
    calbody = parse_calbody(calbody_file_path)
    calreadings = parse_calreadings(calreadings_file_path)
    # for each frame, compute C_i_expected
    C_i_expected_frames = list()
    for frame in calreadings:
        d_vals = calbody["d"]
        a_vals = calbody["a"]
        c_vals = calbody["c"]
        D_vals = frame["D"]
        A_vals = frame["A"]
        F_Dd = pcd_to_pcd_reg_w_known_correspondence(d_vals, D_vals)
        F_Aa = pcd_to_pcd_reg_w_known_correspondence(a_vals, A_vals)
        # compute C_i_expected = F_Dd_inv * F_Aa * c_i
        C_i_expected = F_Dd.inverse_transform_pts(F_Aa.transform_pts(c_vals))
        C_i_expected_frames.append(C_i_expected)
    return np.array(C_i_expected_frames)


def question_4():
    pass
