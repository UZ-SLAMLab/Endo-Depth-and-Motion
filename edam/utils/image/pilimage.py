from typing import List, Optional

import numpy as np
from logzero import logger
from PIL import Image, ImageDraw, ImageFont

from edam.utils.image.convertions import (
    numpy_array_bgr_to_rgb,
    numpy_array_to_pilimage,
    pilimage_to_numpy_array,
)

ALIGNAMENT_H_VALUES = {"L", "C", "R"}
ALIGNAMENT_V_VALUES = {"T", "C", "B"}


def pilimage_h_concat(
    pilimage_list: List[Image.Image],
    alignament_v: Optional[str] = None,
    margin: Optional[int] = 0,
) -> Image.Image:
    """Horizontal concatenation of a list of PIL.Image.

    Arguments:
        pilimage_list {List[Image.Image]} -- Lists of images.

    Keyword Arguments:
        alignament_v {Optional[str]} -- If not all the images have the same H, the
            largest will be used. The smaller ones will be alligned to: {"T", "C", "B"}
            (default: "C")
        margin {Optional[int]} -- Margin between images and in the border, in pixels.
            (default: {0})

    Raises:
        AssertionError: If the provided alignamen is not one of the given options.

    Returns:
        Image.Image -- Concatenated image.
    """
    # -- Default arguments
    alignament_v = alignament_v or "C"
    assert (
        alignament_v in ALIGNAMENT_V_VALUES
    ), f"The value '{alignament_v}' is not implemented, options: {ALIGNAMENT_V_VALUES}."

    # -- Create an blank image.
    concat_w, concat_h = margin, 2 * margin
    for pilimage in pilimage_list:
        w, h = pilimage.size
        concat_w += w + margin
        concat_h = max(concat_h, h + 2 * margin)

    concat_image = Image.new("RGB", (concat_w, concat_h))

    # -- Add all the image as a collage
    idx_w = margin
    for pilimage in pilimage_list:
        w, h = pilimage.size
        if alignament_v == "C":
            idx_h = margin + (concat_h - h) // 2
        elif alignament_v == "T":
            idx_h = margin
        elif alignament_v == "B":
            idx_h = margin + (concat_h - h)
        else:
            raise NotImplementedError(f"Alignament {alignament_v} not implemented.")
        concat_image.paste(pilimage, (idx_w, idx_h))
        idx_w += w + margin

    return concat_image


def pilimage_v_concat(
    pilimage_list: List[Image.Image],
    alignament_h: Optional[str] = None,
    margin: Optional[int] = 0,
) -> Image.Image:
    """Vertical concatenation of a list of PIL.Image.

    Arguments:
        pilimage_list {List[Image.Image]} -- Lists of images.

    Keyword Arguments:
        alignament_h {Optional[str]} -- If not all the images have the same W, the
            largest will be used. The smaller ones will be alligned to: {"L", "C", "R"}
            (default: "L")
        margin {Optional[int]} -- Margin between images and in the border, in pixels.
            (default: {0})

    Raises:
        AssertionError: If the provided alignamen is not one of the given options.

    Returns:
        Image.Image -- Concatenated image.
    """
    # -- Default arguments
    alignament_h = alignament_h or "L"
    assert (
        alignament_h in ALIGNAMENT_H_VALUES
    ), f"The value '{alignament_h}' is not implemented, options: {ALIGNAMENT_H_VALUES}."

    # -- Create an blank image.
    concat_w, concat_h = 2 * margin, margin
    for pilimage in pilimage_list:
        w, h = pilimage.size
        concat_h += h + margin
        concat_w = max(concat_w, w + 2 * margin)

    concat_image = Image.new("RGB", (concat_w, concat_h))

    # -- Add all the image as a collage
    idx_h = margin
    for pilimage in pilimage_list:
        w, h = pilimage.size
        if alignament_h == "C":
            idx_w = margin + (concat_w - w) // 2
        elif alignament_h == "L":
            idx_w = margin
        elif alignament_h == "R":
            idx_w = margin + (concat_w - w)
        else:
            raise NotImplementedError(f"Alignament {alignament_h} not implemented.")
        concat_image.paste(pilimage, (idx_w, idx_h))
        idx_h += h + margin

    return concat_image


def pilimage_rgb_to_bgr(pilimage: Image.Image) -> Image.Image:
    """Transforms an image from RGB to BGR, or the other way arround.
    
    Arguments:
        pilimage {Image.Image} -- [description]
    
    Returns:
        Image.Image -- [description]
    """
    np_image = pilimage_to_numpy_array(pilimage)
    np_image = numpy_array_bgr_to_rgb(np_image)
    return numpy_array_to_pilimage(np_image)


def pilimage_diff(
    pilimage_i: Image.Image, pilimage_j: Image.Image, mask: Optional[Image.Image] = None
) -> Image.Image:
    """Outputs the L2 norm of the difference of two images. Masking to zero the values 
    of the pixels where `mask > 0`.

    Arguments:
        pilimage_i {Image.Image} -- Image 1.
        pilimage_j {Image.Image} -- Image 2.

    Keyword Arguments:
        mask {Optional[Image.Image]} -- Mask (default: {None})

    Returns:
        Image.Image -- Error image.
    """
    image_i = pilimage_to_numpy_array(pilimage_i)
    image_j: np.ndarray.ndarray = pilimage_to_numpy_array(pilimage_j)

    diff_ij = (image_i - image_j) ** 2
    if len(diff_ij.shape) != 2:
        diff_ij = diff_ij.sum(axis=-1) / diff_ij.shape[-1]

    if mask is not None:
        mask = pilimage_to_numpy_array(mask)
        diff_ij[mask > 0] = 0
    return numpy_array_to_pilimage(diff_ij)


def pilimage_add_title(
    pilimage: Image.Image, text: str, v_size_px: int = 14, **kwargs
) -> Image.Image:
    w, h = pilimage.size
    pilimage_w_title = Image.new("RGB", (w, h + v_size_px), color=(0, 0, 0))
    pilimage_w_title.paste(pilimage, (0, v_size_px))

    # Original Images
    try:
        # For Linux
        font_titles = ImageFont.truetype("DejaVuSans.ttf", int(v_size_px / 14.0 * 10.0))
    except Exception:
        logger.warning("No font DejaVuSans; use default instead")
        # For others
        font_titles = ImageFont.load_default()
    draw = ImageDraw.Draw(pilimage_w_title)
    pildraw_text(draw, text, font_titles, w // 2, v_size_px - 1, **kwargs)
    return pilimage_w_title


def pildraw_text(draw, text, font, x, y, color=(255, 255, 255), center=True):
    w, h = draw.textsize(text, font=font)
    xx = (2 * x - w) / 2 if center else x
    yy = y - font.size
    draw.text((xx, yy), text, color, font=font)
    return w
