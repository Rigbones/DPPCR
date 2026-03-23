import numpy as np
import torch

def shuffle(xyz, seed):
    """
    Args:
        xyz (np.ndarray | torch.Tensor): point cloud of shape (N, 3)
        seed (int): random seed for reproducibility
    Returns:
        np.ndarray | torch.Tensor: point cloud of shape (N, 3) with points shuffled
    """
    if (isinstance(xyz, np.ndarray) or isinstance(xyz, torch.Tensor)):
        rng = np.random.default_rng(seed=seed)
        indices = rng.permutation(xyz.shape[0])
        return xyz[indices]
    else:
        raise TypeError(f"Expected np.ndarray or torch.Tensor, but found {type(xyz)}")

def add_noise(xyz, seed):
    """
    Args:
        xyz (np.ndarray | torch.Tensor): point cloud of shape (N, 3)
        seed (int): random seed for reproducibility
    Returns:
        np.ndarray | torch.Tensor: point cloud of shape (N, 3) with 2% Gaussian noise added
    """
    if (isinstance(xyz, np.ndarray)):
        rng = np.random.default_rng(seed=seed)
        mean = np.mean(xyz, axis=0)
        std = np.std(xyz, axis=0)
        size = int(xyz.shape[0] * 0.02)
        noise = rng.normal(loc=mean, scale=std, size=(size, 3))
        return np.concat([xyz, noise], axis=0)
    elif (isinstance(xyz, torch.Tensor)):
        rng = np.random.default_rng(seed=seed)
        mean = torch.mean(xyz, axis=0)
        std = torch.std(xyz, axis=0)
        size = int(xyz.shape[0] * 0.02)
        noise = rng.normal(loc=mean.cpu().numpy(), scale=std.cpu().numpy(), size=(size, 3))
        return torch.concat([xyz, torch.from_numpy(noise).to(device=xyz.device, dtype=torch.float32)], axis=0)
    else:
        raise TypeError(f"Expected np.ndarray or torch.Tensor, but found {type(xyz)}")


def jitter(xyz, seed):
    """ 
    Adds 2% Gaussian jitter to a point cloud.
    Args:
        xyz (np.ndarray | torch.Tensor): point cloud of shape (N, 3)
        seed (int): random seed for reproducibility

    Returns:
        np.ndarray | torch.Tensor: point cloud of shape (N, 3) with Gaussian jitter added
    """
    if (isinstance(xyz, np.ndarray)):
        rng = np.random.default_rng(seed=seed)
        min_std = np.std(xyz, axis=0).min()
        jitter = rng.normal(loc=[0, 0, 0], scale=0.02 * min_std, size=xyz.shape)
        return xyz + jitter
    elif (isinstance(xyz, torch.Tensor)):
        rng = np.random.default_rng(seed=seed)
        min_std = torch.std(xyz, axis=0).min()
        jitter = torch.from_numpy(rng.normal(loc=[0, 0, 0], scale=0.02 * min_std.cpu().numpy(), size=xyz.shape)).to(dtype=xyz.dtype, device=xyz.device)
        return xyz + jitter 
    else:
        raise TypeError(f"Expected np.ndarray or torch.Tensor, but found {type(xyz)}")

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions (torch.tensor | np.ndarray): quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3), as np.ndarray if input is np.ndarray, else torch.Tensor if input is torch.Tensor.
    """
    if (isinstance(quaternions, torch.Tensor)):
        r, i, j, k = torch.unbind(quaternions, -1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)

        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3))
    elif (isinstance(quaternions, np.ndarray)):
        r = quaternions[..., 0]
        i = quaternions[..., 1]
        j = quaternions[..., 2]
        k = quaternions[..., 3]
        two_s = 2.0 / (quaternions * quaternions).sum(-1)

        o = np.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3))
    else:
        raise TypeError(f"Input quaternions should torch.Tensor or np.ndarray, but found {type(quaternions)}")

def axis_angle_to_matrix(axis, angle):
    """Calculates 3x3 rotation matrix from axis-angle."""
    axis = np.asarray(axis) / np.linalg.norm(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = axis
    
    return np.array([
        [t*x*x + c,    t*x*y - s*z,  t*x*z + s*y],
        [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x],
        [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c]
    ])