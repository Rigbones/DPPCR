from pytorch3d.loss import chamfer_distance
import numpy as np
import torch
import copy

from visualize import visualize
from format_conversions import ply_to_np, np_to_ply

def compute_metrics(X, Y, pred, gt):
    """
    Compute rotation, translation, scale and chamfer distance metrics between two point clouds X and Y.
    Translation error is L2 of centroid, scale is absolute difference of scale, chamfer distance uses PyTorch3D's chamfer_distance
    Args:
        X (np.ndarray): source point cloud of shape (N, 3)
        Y (np.ndarray): target point cloud of shape (N, 3)
        pred (np.ndarray): shape (4, 4) predicted homogeneous transformation matrix, where applying pred to X should align with Y
        gt (np.ndarray): shape (4, 4) ground truth homogeneous transformation matrix
    Returns:
        np.ndarray: array of shape (4,) containing 
            - rotation error (degrees), 
            - scale error, 
            - translation error, and 
            - chamfer distance
    """
    # compute scale error 
    S_pred = np.linalg.det(pred[:3, :3]) ** (1/3)
    S_gt = np.linalg.det(gt[:3, :3]) ** (1/3)
    S_error = np.abs(S_pred - S_gt)

    # compute rotation error in degrees
    pred[:3, :3] /= S_pred  # remove scale from rotation part
    gt[:3, :3] /= S_gt # remove scale from rotation part
    R_pred = pred[:3, :3]
    R_gt = gt[:3, :3]
    R_diff = R_pred.T @ R_gt
    trace = np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0)
    R_error = np.arccos(trace) * (180 / np.pi)

    # compute translation error as L2 distance between centroids
    T_pred = pred[:3, 3]
    T_gt = gt[:3, 3]
    T_error = np.linalg.norm(T_pred - T_gt)

    # compute chamfer distance
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(0)
    chamfer_dist = chamfer_distance(X_tensor, Y_tensor)[0].item()

    return np.array([R_error, S_error, T_error, chamfer_dist])

def compute_metrics_one_more_icp(X, Y, pred, gt):
    """
    Same as 'compute_metrics' but applies one more iteration of ICP
    """
    X = (pred[:3, :3] @ X.T).T + pred[:3, 3]
    # apply one iteration of regular ICP
    from tempfile import NamedTemporaryFile
    from subprocess import run
    with NamedTemporaryFile(suffix='.ply') as file1, NamedTemporaryFile(suffix='.ply') as file2, NamedTemporaryFile(suffix='.txt') as out:
        np_to_ply(file1.name, X)
        np_to_ply(file2.name, Y)
        run(['./FRICP', file2.name, file1.name, out.name, '0'])
        icp_pred = np.loadtxt(out.name)

    X = (icp_pred[:3, :3] @ X.T).T + icp_pred[:3, 3]
    visualize([X, Y], ['blue', 'red'], show=False, save="del.png")
    return compute_metrics(X, Y, icp_pred @ pred, gt)


# from utils import axis_angle_to_matrix, jitter, add_noise
# from format_conversions import ply_to_np
# import numpy as np
# from main import random_rigid

# Y = ply_to_np("datasets/smol/sofa_0681.ply")
# seed = int("sofa_0681.ply".split('_')[-1].split('.')[0]) # eg get 0681 from sofa_0681.ply
# X, _ = random_rigid(Y, seed=seed, noise_jitter=False) # get transformed X without the applied transformation matrix

# applied = np.loadtxt('applied.txt')
# pred = np.loadtxt('pred.txt')

# np.set_printoptions(precision=4, suppress=True)
# print(
#     compute_metrics_one_more_icp(X, Y, pred, applied)
# )