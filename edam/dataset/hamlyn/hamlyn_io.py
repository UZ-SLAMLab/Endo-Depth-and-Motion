from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import open3d as o3d
from logzero import logger
from PIL import Image, ImageFile

from edam.utils.errors import DepthReadError
from edam.utils.file import list_files, order_list_paths_by_int_filename
from edam.utils.parser import txt_to_nparray

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_scene_files(scene: str) -> Dict[str, Union[List[str], np.ndarray]]:
    """Loads for a scene the list of files

    Arguments:
        scene {str} -- Path to where the scene is stored.

    Returns:
        Dict[str, Union[List[str], np.ndarray]] -- Output of a dictionary with the keys:
            list_color_images = list of rgb files,
            list_depth_images = list of depth files,
            list_pose_files = list_pose_files,
            extrinsics = extrinsics,
            intrinsics = intrinsics
    """
    # -- Load list of files. (ordered but as string)
    list_color_images = order_list_paths_by_int_filename(
        list_files(str(Path(scene) / "color"))
    )
    list_depth_images = order_list_paths_by_int_filename(
        list_files(str(Path(scene) / "depth"))
    )

    # -- Open intrinsic and extrinsic files.
    if (Path(scene) / "extrinsics.txt").is_file():
        extrinsics = txt_to_nparray(
            open(Path(scene) / "extrinsics.txt")
        )
    intrinsics = txt_to_nparray(
        open(Path(scene) / "intrinsics.txt")
    )

    if (Path(scene) / "extrinsics.txt").is_file():
        return dict(
            list_color_images=list_color_images,
            list_depth_images=list_depth_images,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )
    else:
        return dict(
            list_color_images=list_color_images,
            list_depth_images=list_depth_images,
            intrinsics=intrinsics,
        )


def depth_read(path: str, dtype=np.float16, depth_shift: float = 1000.0) -> np.ndarray:
    """Reads an image from a png file and returns the depth image shifted to meters.
    In Scannet by default the depth_shift is 1000.0

    Arguments:
        path {str} -- Path to the png file.

    Keyword Arguments:
        dtype  --  (default: {np.float16})
        depth_shift {float} -- Value used to divide the values on the png.
            (default: {100.0})

    Returns:
        np.ndarray -- Numpy array cointaining the depth image.
    """
    try:
        depth_image = np.array(Image.open(path)).astype(dtype) / depth_shift
        return depth_image
    except BaseException as e:
        Image.open(path).show()
        logger.error(f'Error when loading image: "{path}"')
        logger.error(e)
        raise DepthReadError(e)


def load_pointcloud_reconstruction(path: str):
    """Load a point cloud.

    Arguments:
        path {str} -- [description]

    Returns:
        [type] -- [description]
    """
    pcd = o3d.io.read_point_cloud(path)
    return pcd


# --------------------------------------------------------------------------------------
# Private helpers
