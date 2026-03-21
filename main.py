import torch

from utils import axis_angle_to_matrix, jitter, add_noise
from format_conversions import ply_to_np
import numpy as np
import copy
from trainer import DP_PCR

if __name__ == "__main__":
    X = ply_to_np("datasets/dresser_0201.ply") / 10 # shape (N, 3)
    Y = copy.deepcopy(X)

    Y = Y - Y.mean(axis=0) # shape (N, 3)
    X = X - X.mean(axis=0)

    rng = np.random.default_rng(seed=3043)
    # get ground truth transformation
    GT_R1 = axis_angle_to_matrix([1, 0, 0], np.radians(rng.uniform(0, 40)))
    GT_R2 = axis_angle_to_matrix([0, 1, 0], np.radians(rng.uniform(0, 40)))
    GT_R3 = axis_angle_to_matrix([0, 0, 1], np.radians(rng.uniform(0, 40)))
    GT_S = np.array([rng.uniform(0.5, 0.7) if rng.random() < 0.5 else rng.uniform(1.3, 1.5)])
    GT_T = rng.uniform(0.5, 0.5, size=3)

    # apply transformations to X
    GT = np.eye(4)
    GT[:3, :3] = GT_R3 @ GT_R2 @ GT_R1
    GT[:3, :3] = GT[:3, :3] * GT_S
    GT[:3, 3] = GT_T

    X = GT_S * (GT[:3, :3] @ X.T).T + GT[:3, 3]
    X = add_noise(jitter(X, seed=3043), seed=3043)

    # X = add_noise(jitter(X))
    print(GT)

    DP_PCR(X, Y, device='cuda:0')