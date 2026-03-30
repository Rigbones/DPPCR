# python libraries
import numpy as np
from subprocess import run, DEVNULL
from tempfile import NamedTemporaryFile
import os
import pickle
from time import perf_counter
import argparse

# my functions
from format_conversions import ply_to_np, np_to_ply, np_to_xyz
from trainer import DP_PCR
from metrics import compute_metrics
from utils import shuffle, add_noise, jitter, axis_angle_to_matrix
from visualize import visualize

def random_rigid(X, seed, noise_jitter_shuffle_downsample=True):
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
    if (noise_jitter_shuffle_downsample):
        X = shuffle(add_noise(jitter(X, seed=seed), seed=seed), seed=seed)[::2]

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
        7: SA-ICP
        8: No Bounding Box ICP
    """
    def X_to_Y_scale_factor(X, Y):
        mins_X = np.min(X, axis=0)
        maxs_X = np.max(X, axis=0)
        mins_Y = np.min(Y, axis=0)
        maxs_Y = np.max(Y, axis=0)
        diameter_X = np.linalg.norm(maxs_X - mins_X)
        diameter_Y = np.linalg.norm(maxs_Y - mins_Y)
        return diameter_Y / diameter_X

    if (method not in [0, 1, 3, 6, 7, 8]):
        raise ValueError(f"Invalid method {method}. Must be one of [0, 1, 3, 6, 7, 8]")

    if (method in [0, 1, 3, 6]): # run ICP / AAICP / FRICP / Sparse ICP
        # do a simple bounding box adjustment: scale until bounding boxes of each has diameter 1
        scale_factor = X_to_Y_scale_factor(X, Y)
        scale_mat = np.eye(4)
        scale_mat[:3, :3] = np.eye(3) * scale_factor
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
            run(['./SAICP', in1.name, in2.name, out.name, "1", "1", "50"], stdout=DEVNULL) # method 1 for SA-ICP
            pred_pcd = np.loadtxt(out.name)
            pred = get_rigid_between_pcd(X, pred_pcd) 
        return pred

    elif (method == 8): # no bounding box ICP
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


M0_metrics = {}
M1_metrics = {}
M3_metrics = {}
M6_metrics = {}
M7_metrics = {}
Mours_metrics = {}

if __name__ == "__main__":
    # use argparse to optionally specify which half of the dataset to run on (first_half or second_half)
    parser = argparse.ArgumentParser(description='Test DPPCR vs other methods')
    parser.add_argument('--clean', 
                        required=True, type=int, choices=[0, 1], help='0 for noise + jitter + shuffle + downsample augmentation, 1 for clean data')
    parser.add_argument('--portion', 
                        required=True, type=int, choices=[0, 1], help='0 to run on first half, 1 to run on second half')
    parser.add_argument('--methods', 
                        required=True, type=int, nargs='+', choices=[0, 1, 3, 6, 7, 10], help='Which methods to run. 0: ICP, 1: AAICP, 3: FRICP, 6: Sparse ICP, 7: SA-ICP, 10: Ours')
    parser.add_argument('--visualize', 
                        required=False, type=int, choices=[0, 1], default=0, help='Whether to save visualizations to figs folder')
    parser.add_argument('--device',
                        required=False, type=int, choices=[0, 1], help='Which GPU to run on')
    args = parser.parse_args()
    
    # loop through datasets/smol/, apply random rigid transformation, run different methods and compute metrics
    filenames = list(os.listdir("datasets/smol/"))
    filenames.sort()
    filenames = filenames[:450] if args.portion == 0 else filenames[450:]

    for name in filenames:
        seed = int(name.split('_')[-1].split('.')[0]) # eg get 0681 from sofa_0681.ply

        Y = ply_to_np("datasets/smol/" + name)
        X, applied = random_rigid(Y, seed=seed, noise_jitter_shuffle_downsample=args.clean == 0)
        X_clean, _ = random_rigid(Y, seed=seed, noise_jitter_shuffle_downsample=False)

        print("--------------")
        print(f"[{filenames.index(name) + 1}/{len(filenames)}] Running '{name}', part of portion {args.portion}, methods {args.methods} with {'clean data' if args.clean == 1 else 'augmented data'}")

        if (0 in args.methods): # run ICP
            start = perf_counter()
            pred = run_others(X, Y, method=0)
            elapsed = perf_counter() - start
            metrics = np.append(compute_metrics(X_clean, Y, pred, applied), elapsed)
            M0_metrics[name] = metrics
            if (args.visualize == 1):
                visualize([(pred[:3, :3] @ X_clean.T).T + pred[:3, 3], Y], ['blue', 'red'], show=False, save=f"figs/M0.png")

        if (1 in args.methods): # run AAICP
            start = perf_counter()
            pred = run_others(X, Y, method=1)
            elapsed = perf_counter() - start
            metrics = np.append(compute_metrics(X_clean, Y, pred, applied), elapsed)
            M1_metrics[name] = metrics
            if (args.visualize == 1):
                visualize([(pred[:3, :3] @ X_clean.T).T + pred[:3, 3], Y], ['blue', 'red'], show=False, save=f"figs/M1.png")

        if (3 in args.methods): # run FRICP
            start = perf_counter()
            pred = run_others(X, Y, method=3)
            elapsed = perf_counter() - start
            metrics = np.append(compute_metrics(X_clean, Y, pred, applied), elapsed)
            M3_metrics[name] = metrics
            if (args.visualize == 1):
                visualize([(pred[:3, :3] @ X_clean.T).T + pred[:3, 3], Y], ['blue', 'red'], show=False, save=f"figs/M3.png")

        if (6 in args.methods): # run Sparse ICP
            start = perf_counter()
            pred = run_others(X, Y, method=6)
            elapsed = perf_counter() - start
            metrics = np.append(compute_metrics(X_clean, Y, pred, applied), elapsed)
            M6_metrics[name] = metrics
            if (args.visualize == 1):
                visualize([(pred[:3, :3] @ X_clean.T).T + pred[:3, 3], Y], ['blue', 'red'], show=False, save=f"figs/M6.png")

        if (7 in args.methods): # run SA-ICP
            start = perf_counter()
            pred = run_others(X, Y, method=7)
            elapsed = perf_counter() - start
            metrics = np.append(compute_metrics(X_clean, Y, pred, applied), elapsed)
            M7_metrics[name] = metrics
            if (args.visualize == 1):
                visualize([(pred[:3, :3] @ X_clean.T).T + pred[:3, 3], Y], ['blue', 'red'], show=False, save=f"figs/M7.png")

        
        if (10 in args.methods): # run ours
            start = perf_counter()
            pred = DP_PCR(X, Y, device=f'cuda:{args.portion}' if args.device is None else f'cuda:{args.device}')
            pred2 = run_others((pred[:3, :3] @ X.T).T + pred[:3, 3], Y, method=8)
            elapsed = perf_counter() - start
            pred = pred2 @ pred
            metrics = compute_metrics(X_clean, Y, pred, applied)
            Mours_metrics[name] = np.append(metrics, elapsed)
            if (args.visualize == 1):
                visualize([(pred[:3, :3] @ X_clean.T).T + pred[:3, 3], Y], ['blue', 'red'], show=False, save=f"figs/Mours.png")


        # save the dictionaries using pickle
        pkl_name = "first" if args.portion == 0 else "second"
        if (0 in args.methods):
            with open(f'results/M0_{pkl_name}.pkl', 'wb') as f:
                pickle.dump(M0_metrics, f)
        if (1 in args.methods):
            with open(f'results/M1_{pkl_name}.pkl', 'wb') as f:
                pickle.dump(M1_metrics, f)
        if (3 in args.methods):
            with open(f'results/M3_{pkl_name}.pkl', 'wb') as f:
                pickle.dump(M3_metrics, f)
        if (6 in args.methods):
            with open(f'results/M6_{pkl_name}.pkl', 'wb') as f:
                pickle.dump(M6_metrics, f)
        if (7 in args.methods):
            with open(f'results/M7_{pkl_name}.pkl', 'wb') as f:
                pickle.dump(M7_metrics, f)
        if (10 in args.methods):
            with open(f'results/Mours_{pkl_name}.pkl', 'wb') as f:
                pickle.dump(Mours_metrics, f)

        