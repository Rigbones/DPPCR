from plotly import graph_objects as go
import numpy as np

def visualize(xyz, rgb=None, marker_size=0.5, axes=False, show=True, save=None, forceplot=False):
    """
    Args:
        xyz (np.ndarray | list[np.ndarray]): point cloud(s) of shape (N, 3)
        rgb (np.ndarray | list[np.ndarray] | str | list[str] | None): shape (N, 3) in np.uint8, or string for color names.  Currently supports 'blue', 'red', 'green', 'black'.
        marker_size (float): size of each point
        axes (bool): whether to show x, y, z axes in the plot
        show (bool): whether to show the plot in Jupyter notebook
        save (str | None): if not None, saves an image of the plot to the path given by the value of the parameter

    Usage:
        visualize([X, Y], ['blue', 'red'])
    """
    # convert to list 
    if isinstance(xyz, np.ndarray):
        xyz = [xyz]
    if rgb is None:
        rgb = ['black' for _ in range(len(xyz))]
    if isinstance(rgb, np.ndarray) or isinstance(rgb, str):
        rgb = [rgb]

    # if too many points, quit 
    if not forceplot:
        for arr in xyz:
            if arr.shape[0] > 50_000:
                print(f"Too many points ({arr.shape[0]})")
                return

    _rgb = []
    colormap = {
        'blue': np.array([0, 0, 255], dtype=np.uint8),
        'red': np.array([255, 0, 0], dtype=np.uint8),
        'green': np.array([100, 255, 100], dtype=np.uint8),
        'black': np.array([10, 10, 10], dtype=np.uint8)
    }

    # loop through xyz list to construct colors
    for i, j in zip(xyz, rgb):
        # if rgb is given as np array, just append it to _rgb
        if (isinstance(j, np.ndarray)):
            _rgb.append(j)
        elif (isinstance(j, str)):
            _rgb.append(
                np.tile(colormap[j], (i.shape[0], 1))
            )

    # concatenate all xyz and rgb
    xyz = np.concat(xyz, axis=0)
    rgb = np.concat(_rgb, axis=0)
    if (save is None):
        print(f"Visualizing a NumPy array of shape {xyz.shape}, rgb of shape {rgb.shape}")

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xyz[:,0], y=xyz[:,1], z=xyz[:,2], 
                mode='markers',
                marker=dict(size=marker_size, color=rgb.astype(np.uint8))
            )
        ],
        layout=dict(
            scene=dict(
                xaxis=dict(visible=axes),
                yaxis=dict(visible=axes),
                zaxis=dict(visible=axes)
            )
        )
    )

    if show:
        fig.show()

    if (save is not None):
        fig.write_image(save, scale=2.0)
        print(f"Saved to {save}")
