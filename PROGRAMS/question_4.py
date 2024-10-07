import pdb

# from utils import pcd_to_pcd_reg
from test import icp as pcd_to_pcd_reg

from data_processing import parse_calbody, parse_calreadings


def main():
    """The main function for Programming Assignment 1, question 4."""
    # Question 4a : compute F_D for each frame
    # read relevant data
    calbody = parse_calbody("../DATA/pa1-unknown-k-calbody.txt")
    calreadings_frames = parse_calreadings("../DATA/pa1-unknown-k-calreadings.txt")
    # for each frame, compute F_D between the optical tracker and EM tracker coords
    F_D_frames = list()
    for frame in calreadings_frames:
        d_vals = calbody["d"]
        D_vals = frame["D"]
        assert d_vals.shape == D_vals.shape
        F_D = pcd_to_pcd_reg(D_vals, d_vals)
        print(d_vals - F_D.transform_pts(D_vals))
        print(D_vals - F_D.transform_pts(d_vals))
        print(D_vals - F_D.inverse_transform_pts(d_vals))
        F_D_frames.append(F_D)
    # Question 4b : compute F_A for each frame
    F_A_frames = list()
    for frame in calreadings_frames:
        a_vals = calbody["a"]
        A_vals = frame["A"]
        assert a_vals.shape == A_vals.shape
        F_A = pcd_to_pcd_reg(A_vals, a_vals)
        F_A_frames.append(F_A)
    # Question 4c : compute C_expected for each frame


if __name__ == "__main__":
    main()
