from typing import Dict, List

import numpy as np

dataset_prefixes = [
    "pa1-debug-a-",
    "pa1-debug-b-",
    "pa1-debug-c-",
    "pa1-debug-d-",
    "pa1-debug-e-",
    "pa1-debug-f-",
    "pa1-debug-g-",
    "pa1-unknown-h-",
    "pa1-unknown-i-",
    "pa1-unknown-j-",
    "pa1-unknown-k-",
]


def parse_calbody(path: str) -> Dict:
    """Parses a calbody.txt file according to the specifications
    in the homework description.

    Params:
        path (str): file path to the dataset file

    Returns:
        Dict: dictionary containing the d, a, and c values
    """
    assert "calbody" in path, "Wrong file."
    with open(path, "r") as file:
        data = file.readlines()
    for idx, line in enumerate(data):
        data[idx] = line.replace(",", "")
    N_D, N_A, N_C, _ = data[0].split(" ")
    N_D, N_A, N_C = int(N_D), int(N_A), int(N_C)
    idx = 1
    d = list()
    a = list()
    c = list()
    for i in range(N_D):
        d.append([float(x) for x in data[idx + i].split(" ") if x != ""])
    idx += N_D
    for i in range(N_A):
        a.append([float(x) for x in data[idx + i].split(" ") if x != ""])
    idx += N_A
    for i in range(N_C):
        c.append([float(x) for x in data[idx + i].split(" ") if x != ""])
    return {"d": np.array(d), "a": np.array(a), "c": np.array(c)}


def parse_calreadings(path: str) -> List[Dict]:
    """Parses a calreadings.txt file according to the specifications
    in the homework description.

    Params:
        path (str): file path to the dataset file

    Returns:
        List

    """
    assert "calreadings" in path, "Wrong file."
    with open(path, "r") as file:
        data = file.readlines()
    for idx, line in enumerate(data):
        data[idx] = line.replace(",", "")
    N_D, N_A, N_C, N_frames, _ = data[0].split(" ")
    N_D, N_A, N_C, N_frames = int(N_D), int(N_A), int(N_C), int(N_frames)
    idx = 1
    frames = list()
    for _ in range(N_frames):
        D = list()
        A = list()
        C = list()
        for i in range(N_D):
            D.append([float(x) for x in data[idx + i].split(" ") if x != ""])
        idx += N_D
        for i in range(N_A):
            A.append([float(x) for x in data[idx + i].split(" ") if x != ""])
        idx += N_A
        for i in range(N_C):
            C.append([float(x) for x in data[idx + i].split(" ") if x != ""])
        idx += N_C
        frames.append({"D": np.array(D), "A": np.array(A), "C": np.array(C)})
    return frames


def parse_empivot(path: str) -> List[Dict]:
    """Parses a empivot.txt file according to the specifications
    in the homework description."""
    assert "empivot" in path, "Wrong file."
    with open(path, "r") as file:
        data = file.readlines()
    for idx, line in enumerate(data):
        data[idx] = line.replace(",", "")
    N_G, N_frames, _ = data[0].split(" ")
    N_G, N_frames = int(N_G), int(N_frames)
    idx = 1
    frames = list()
    for _ in range(N_frames):
        G = list()
        for i in range(N_G):
            G.append([float(x) for x in data[idx + i].split(" ") if x != ""])
        frames.append({"G": np.array(G)})
        idx += N_G
    return frames


def parse_optpivot(path: str) -> List[Dict]:
    """Parses a empivot.txt file according to the specifications
    in the homework description."""
    assert "optpivot" in path, "Wrong file."
    with open(path, "r") as file:
        data = file.readlines()
    for idx, line in enumerate(data):
        data[idx] = line.replace(",", "")
    N_D, N_H, N_frames, _ = data[0].split(" ")
    N_D, N_H, N_frames = int(N_D), int(N_H), int(N_frames)
    idx = 1
    frames = list()
    for _ in range(N_frames):
        D = list()
        H = list()
        for i in range(N_D):
            D.append([float(x) for x in data[idx + i].split(" ") if x != ""])
        idx += N_D
        for i in range(N_H):
            H.append([float(x) for x in data[idx + i].split(" ") if x != ""])
        frames.append({"D": np.array(D), "H": np.array(H)})
        idx += N_H
    return frames


if __name__ == "__main__":
    path = (
        "/Users/byrdgb1/Desktop/Classes/CIS_1/CIS_1_PA_1/DATA/pa1-unknown-k-calbody.txt"
    )
    parse_calbody(path)
    path = "/Users/byrdgb1/Desktop/Classes/CIS_1/CIS_1_PA_1/DATA/pa1-unknown-k-calreadings.txt"
    parse_calreadings(path)
    path = (
        "/Users/byrdgb1/Desktop/Classes/CIS_1/CIS_1_PA_1/DATA/pa1-unknown-k-empivot.txt"
    )
    parse_empivot(path)
    path = "/Users/byrdgb1/Desktop/Classes/CIS_1/CIS_1_PA_1/DATA/pa1-unknown-k-optpivot.txt"
    parse_optpivot(path)
