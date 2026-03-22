import open3d as o3d
from plyfile import PlyData
import numpy as np

def off_to_ply(off_path: str, ply_path: str, num_pts: int = 30_000):
    """
    Converts an OFF mesh file to an ascii PLY point cloud by sampling points on the mesh surface.
    
    Args:
        off_path (str): Path to the OFF file.
        ply_path (str): Path to save the PLY file.
        num_pts (int): Number of points to sample on the mesh surface.
    """
    mesh = o3d.io.read_triangle_mesh(off_path)
    pcd = mesh.sample_points_uniformly(number_of_points=density)
    o3d.io.write_point_cloud(ply_path, pcd, write_ascii=True)

def np_to_ply(ply_path: str, xyz: np.ndarray):
    """
    Saves a NumPy array of shape (N, 3) as a PLY point cloud.
    No color information is saved.

    Args:
        ply_path (str): path to save the PLY file
        xyz (np.ndarray): point cloud of shape (N, 3)
    """
    num_pts = xyz.shape[0]

    # header for ply file
    header = "\n".join([
        "ply",
        "format ascii 1.0",
        f"element vertex {num_pts}",
        "property float x",
        "property float y",
        "property float z",
        "end_header\n"
    ])

    # create what to write
    contents = []
    for i in range(0, xyz.shape[0]):
        x, y, z = xyz[i]
        contents.append(f'{x} {y} {z}\n')

    # write content to ply
    with open(ply_path, 'w') as f:
        f.write(header)
        f.writelines(contents)

def ply_to_np(ply_path: str):
    """ Loads a PLY point cloud as a NumPy array of shape (N, 3) """
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data
    return np.stack([v["x"], v["y"], v["z"]], axis=1)
