from typing import Optional

import numpy as np
from matplotlib import pyplot as plt


def depth_to_color(
    depth_np: np.ndarray,
    cmap: str = "gist_rainbow",
    max_depth: Optional[float] = None,
    min_depth: Optional[float] = None,
) -> np.ndarray:
    """Converts a depth image to color using the specified color map.

    Arguments:
        depth_np {np.ndarray} -- Depth image/Or inverse depth image. [HxW]
    Keyword Arguments:
        cmap {str} -- Color map to be used. (default: {"gist_rainbow"})

    Returns:
        np.ndarray -- Color image [HxWx3]
    """
    # -- Set default arguments
    depth_np[np.isinf(depth_np)] = np.nan
    if max_depth is None:
        max_depth = np.nanmax(depth_np)
    if min_depth is None:
        min_depth = min_depth or np.nanmin(depth_np)

    cm = plt.get_cmap(cmap, lut=1000)
    depth_np_norm = (depth_np - min_depth) / (max_depth - min_depth)
    colored_depth = cm(depth_np_norm)

    return (colored_depth[:, :, :3] * 255).astype(np.uint8)
