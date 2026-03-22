from utils import axis_angle_to_matrix, jitter, add_noise
from format_conversions import ply_to_np
import numpy as np
import copy
from trainer import DP_PCR

def random_rigid(X, seed, noise_jitter=True):
    """
    Applies a random rigid transformation to a point cloud X, along with noise and jitter.
    Args:
        X (np.ndarray): point cloud of shape (N, 3) to be transformed
        seed (int): random seed for reproducibility

    Returns:
        Tuple[np.ndarray, np.ndarray]: transformed point cloud of shape (N, 3) and the applied transformation matrix of shape (4, 4)
    """
    rng = np.random.default_rng(seed=seed)
    # get random transformation
    R1 = axis_angle_to_matrix([1, 0, 0], np.radians(rng.uniform(0, 40))) # (3, 3)
    R2 = axis_angle_to_matrix([0, 1, 0], np.radians(rng.uniform(0, 40))) # (3, 3)
    R3 = axis_angle_to_matrix([0, 0, 1], np.radians(rng.uniform(0, 40))) # (3, 3)
    S = np.array([rng.uniform(0.3, 0.7) if rng.random() < 0.5 else rng.uniform(1.3, 1.7)]) # (1,)
    T = rng.uniform(-2.0, 2.0, size=3) # (3,)

    # apply transformations, noise, jitter to X
    X = (np.linalg.inv(R1 @ R2 @ R3) @ ((1 / S) * (X - T)).T).T
    if (noise_jitter):
        X = add_noise(jitter(X, seed=seed), seed=seed)

    # save applied transformation to calculate DP-PCR's accuracy
    applied =  np.eye(4)
    applied[:3, :3] = R1 @ R2 @ R3
    applied[:3, :3] = applied[:3, :3] * S
    applied[:3, 3] = T

    return X, applied

if __name__ == "__main__":
    ply = "datasets/smol/sofa_0681.ply"
    Y = ply_to_np(ply)
    print(Y[0:2])

    seed = int(ply.split('_')[-1].split('.')[0]) # eg get 0681 from sofa_0681.ply
    X, applied = random_rigid(Y, seed=seed)

    np.savetxt('applied.txt', applied)
    print(Y[0:2])
    print(X[0:2])

    # DP_PCR(X, Y, device='cuda:0')