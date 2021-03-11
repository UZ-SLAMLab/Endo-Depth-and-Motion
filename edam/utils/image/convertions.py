from typing import Optional

import numpy as np
from PIL import Image


def numpy_array_to_pilimage(
    np_image: np.ndarray, mode: Optional[str] = None, ha=4, h=4
) -> Image.Image:
    """Converts a numpy array to a PIL.Image object.

    Arguments:
        np_image {np.ndarray} -- Shape (h, w) or (h, w, c) for c > 1.

    Keyword Arguments:
        mode {Optional[str]} -- Mode to use (will be determined from type if None)
            See: [Modes](https://pillow.readthedocs.io/en/3.1.x/handbook/concepts.html#concept-modes). # noqa: E501
            (default: {None})

    Returns:
        Image.Image -- Returnde PIL.Image object.
    """
    return Image.fromarray(np_image, mode=mode)


def pilimage_to_numpy_array(pilimage: Image.Image) -> np.ndarray:
    """Converts a PIL.Image to numpy array.

    Arguments:
        pilimage {Image.Image} -- Given image to convert.

    Returns:
        np.ndarray -- Returned array
    """
    return np.array(pilimage)


def numpy_array_bgr_to_rgb(np_image: np.ndarray) -> np.ndarray:
    """Transforms an image from RGB to BGR, or the other way arround.
    
    Arguments:
        np_image {np.ndarray} -- Image in array format of shape (h, w, c)
    
    Returns:
        np.ndarray -- Image in array format transformed
    """
    return np_image[..., ::-1]
