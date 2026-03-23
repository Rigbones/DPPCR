# python libraries
import numpy as np
from subprocess import run, DEVNULL
from tempfile import NamedTemporaryFile

# my functions
from format_conversions import ply_to_np, np_to_ply, np_to_xyz
from trainer import DP_PCR
from metrics import compute_metrics
from utils import shuffle, add_noise, jitter, axis_angle_to_matrix
from visualize import visualize

def random_rigid(X, seed, noise_jitter_shuffle=True):
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
    if (noise_jitter_shuffle):
        X = shuffle(add_noise(jitter(X, seed=seed), seed=seed), seed=seed)

    # save applied transformation to calculate DP-PCR's accuracy
    applied =  np.eye(4)
    applied[:3, :3] = R1 @ R2 @ R3
    applied[:3, :3] = applied[:3, :3] * S
    applied[:3, 3] = T

    return X, applied

def run_others(X, Y, method):
    """Run others
    method:
        0: ICP
        1: AAICP
        3: FRICP
        6: Sparse ICP
        7: SA-ICP (not implemented yet)
    """
    def X_to_Y_scale_factor(X, Y):
        mins_X = np.min(X, axis=0)
        maxs_X = np.max(X, axis=0)
        mins_Y = np.min(Y, axis=0)
        maxs_Y = np.max(Y, axis=0)
        diameter_X = np.linalg.norm(maxs_X - mins_X)
        diameter_Y = np.linalg.norm(maxs_Y - mins_Y)
        return diameter_Y / diameter_X

    if (method not in [0, 1, 3, 6, 7, -1]):
        raise ValueError(f"Invalid method {method}. Must be one of [0, 1, 3, 6, 7]")

    if (method in [0, 1, 3, 6]): # run ICP / AAICP / FRICP / Sparse ICP
        # do a simple bounding box adjustment: scale until bounding boxes of each has diameter 1
        scale_factor = X_to_Y_scale_factor(X, Y)
        scale_mat = np.eye(4)
        scale_mat[:3, :3] = np.eye(3) * scale_factor
        print(scale_mat)
        X = X * scale_factor

        with NamedTemporaryFile(suffix='.ply') as in1, NamedTemporaryFile(suffix='.ply') as in2, NamedTemporaryFile(suffix='.txt') as out:
            # in1 is source, in2 is target, out is (4, 4) rigid transformation txt file
            np_to_ply(in1.name, X)
            np_to_ply(in2.name, Y)
            # Usage: target.ply source.ply out_path <Method>
            run(['./FRICP', in2.name, in1.name, out.name, str(method)], stdout=DEVNULL)
            pred = np.loadtxt(out.name)
        return pred @ scale_mat

    elif (method == 7): # run SA-ICP
        with NamedTemporaryFile(suffix='.xyz') as in1, NamedTemporaryFile(suffix='.xyz') as in2, NamedTemporaryFile(suffix='.txt') as out:
            # in1 is source, in2 is target, out is (4, 4) rigid transformation txt file
            np_to_xyz(in1.name, X)
            np_to_xyz(in2.name, Y)
            # Usage: source.xyz target.xyz dest <method> <1-1> <nMaxIterations> <minDisplacement: optional -- default 0.001>
            run(['./SAICP', in1.name, in2.name, out.name, "1", "1", "20"]) # method 1 for SA-ICP
            pred_pcd = np.loadtxt(out.name)
            print(X.shape, pred_pcd.shape)
            pred = get_rigid_between_pcd(X, pred_pcd) 
        return pred

    elif (method == -1): # debug method
        # do a simple bounding box adjustment: scale until bounding boxes of each has diameter 1
        with NamedTemporaryFile(suffix='.ply') as in1, NamedTemporaryFile(suffix='.ply') as in2, NamedTemporaryFile(suffix='.txt') as out:
            # in1 is source, in2 is target, out is (4, 4) rigid transformation txt file
            np_to_ply(in1.name, X)
            np_to_ply(in2.name, Y)
            # Usage: target.ply source.ply out_path <Method>
            run(['./FRICP', in2.name, in1.name, out.name, "0"], stdout=DEVNULL)
            pred = np.loadtxt(out.name)
        return pred 

def get_rigid_between_pcd(A, B):
    """
    Estimates the rigid transformation (Rotation, Scale, Translation) that aligns A to B using SVD,
    assuming each A[i] corresponds to B[i]

    Args:
        A (np.ndarray): Source points, shape (N, 3)
        B (np.ndarray): Target points, shape (N, 3)

    Returns:
        np.ndarray: Homogeneous transform T (4, 4) such that B = SR(A) + T
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != 3 or B.shape[1] != 3:
        raise ValueError(f"A and B must both have shape (N, 3), got {A.shape} and {B.shape}")
    if A.shape[0] != B.shape[0]:
        raise ValueError(f"A and B must have the same number of points, got {A.shape[0]} and {B.shape[0]}")
    if A.shape[0] < 3:
        raise ValueError("Need at least 3 corresponding points to estimate a 3D transform")

    n = A.shape[0]

    # Centroids
    mu_A = A.mean(axis=0)
    mu_B = B.mean(axis=0)

    # Centered point clouds
    A0 = A - mu_A
    B0 = B - mu_B

    # Cross-covariance (target <- source)
    Sigma = (B0.T @ A0) / n  # (3, 3)

    # SVD
    U, D, Vt = np.linalg.svd(Sigma)

    # Reflection handling to ensure proper rotation (det = +1)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt

    # Uniform scale
    var_A = np.mean(np.sum(A0 ** 2, axis=1))
    if var_A <= np.finfo(np.float64).eps:
        raise ValueError("Degenerate source point cloud: variance is too small to estimate scale")
    scale = np.trace(np.diag(D) @ S) / var_A

    # Translation
    t = mu_B - scale * (R @ mu_A)

    # Homogeneous transform
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = scale * R
    T[:3, 3] = t

    return T

if __name__ == "__main__":
    ply = "datasets/smol/monitor_0466.ply"
    seed = int(ply.split('_')[-1].split('.')[0]) # eg get 0681 from sofa_0681.ply

    Y = ply_to_np(ply)
    X, applied = random_rigid(Y, seed=seed)
    X_clean, _ = random_rigid(Y, seed=seed, noise_jitter_shuffle=False)

    # run debug
    # pred = run_others(X, Y, method=-1)
    # metrics = compute_metrics(X_clean, Y, pred, applied)
    # print(metrics)
    # visualize([(pred[:3, :3] @ X_clean.T).T + pred[:3, 3], Y], ['blue', 'red'], show=False, save=f"figs/Mdebug.png")

    # run ICP
    pred = run_others(X, Y, method=0)
    metrics = compute_metrics(X_clean, Y, pred, applied)
    print(metrics)
    visualize([(pred[:3, :3] @ X_clean.T).T + pred[:3, 3], Y], ['blue', 'red'], show=False, save=f"figs/M0.png")
    
    # run AAICP
    pred = run_others(X, Y, method=1)
    metrics = compute_metrics(X_clean, Y, pred, applied)
    print(metrics)
    visualize([(pred[:3, :3] @ X_clean.T).T + pred[:3, 3], Y], ['blue', 'red'], show=False, save=f"figs/M1.png")

    # run FRICP
    pred = run_others(X, Y, method=3)
    metrics = compute_metrics(X_clean, Y, pred, applied)
    print(metrics)
    visualize([(pred[:3, :3] @ X_clean.T).T + pred[:3, 3], Y], ['blue', 'red'], show=False, save=f"figs/M3.png")

    # run Sparse ICP
    pred = run_others(X, Y, method=6)
    metrics = compute_metrics(X_clean, Y, pred, applied)
    print(metrics)
    visualize([(pred[:3, :3] @ X_clean.T).T + pred[:3, 3], Y], ['blue', 'red'], show=False, save=f"figs/M6.png")

    # run SA-ICP
    pred = run_others(X, Y, method=7)
    metrics = compute_metrics(X_clean, Y, pred, applied)
    print(metrics)
    visualize([(pred[:3, :3] @ X_clean.T).T + pred[:3, 3], Y], ['blue', 'red'], show=False, save=f"figs/M7.png")
    
    # run ours
    pred = DP_PCR(X, Y)
    pred2 = run_others((pred[:3, :3] @ X.T).T + pred[:3, 3], Y, method=0)
    pred = pred2 @ pred
    metrics = compute_metrics(X_clean, Y, pred, applied)
    print(metrics)
    visualize([(pred[:3, :3] @ X_clean.T).T + pred[:3, 3], Y], ['blue', 'red'], show=False, save=f"figs/Mours.png")

