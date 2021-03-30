import kornia
import numpy as np
import torch

from edam.geometry.depth_to_flow import _backproject_points, _transform_points
from .utils import _plot_real_synth_error


def photometric_error_gen(
    i_pose_w: torch.Tensor,
    depth_ref: torch.Tensor,
    k_ref: torch.Tensor,
    j_pose_w: torch.Tensor,
    k_j_target: torch.Tensor,
    gray_i: torch.Tensor,
    gray_j: torch.Tensor,
    margins: int = 12,
    saturation: int = 254,
    plot: bool = False,
) -> np.array:
    """
    Estimate the photometric error of the back-projection of the
    first frame observed in a second frame.


    Args:
        i_pose_w: torch.tensor, # [B_refx4x4]
        depth_ref: torch.tensor, # [B_refx1xHxW]
        K_ref: torch.tensor, # [1x3x3]
        j_pose_w: torch.tensor, # [Bx4x4]
        k_target: torch.tensor, # [Bx3x3]
        gray_ref: torch.tensor, # [B_refxHxW]
        gray_target: torch.tensor, # [BxHxW]
        plot: bool, plot image. Just for debugging.
    Returns:
        error_ij # [BxHxW]
    """
    i_points_3d_i = _backproject_points(
        depth_ref, k_ref, normalize_points=False,
    )  # NxHxWx3

    i_points_3d_j = _transform_points(
        i_points_3d_i.float(), i_pose_w.float(), j_pose_w.float(),
    )  # NxHxWx3

    i_points_2d_j: torch.Tensor = i_points_3d_j.new_zeros(
        (i_points_3d_j.shape[0], i_points_3d_j.shape[1], i_points_3d_j.shape[2], 2)
    )
    for j in range(0, i_points_3d_j.shape[0]):
        i_points_2d_j[j, ...] = kornia.project_points(
            i_points_3d_j[j, ...], k_j_target[j, ...]
        )

    height, width = gray_j.shape[-2:]
    i_points_2d_j_norm: torch.Tensor = kornia.normalize_pixel_coordinates(
        i_points_2d_j, height, width
    ).double()

    image_j_in_i = torch.nn.functional.grid_sample(
        gray_j.double(), i_points_2d_j_norm, align_corners=True,  # type: ignore
    )
    # Not take into account the black in the gray image of the first keyframe
    black_acceptance = gray_i > 0

    # Default margins 12.
    width_acceptance = torch.bitwise_and(
        i_points_2d_j[..., 0] >= margins,
        i_points_2d_j[..., 0] < gray_j.shape[-1] - margins,
    ).unsqueeze(1)
    height_acceptance = torch.bitwise_and(
        i_points_2d_j[..., 1] >= margins,
        i_points_2d_j[..., 1] < gray_j.shape[-2] - margins,
    ).unsqueeze(1)
    projected_acceptance = torch.bitwise_and(
        black_acceptance, torch.bitwise_and(width_acceptance, height_acceptance)
    )

    error = error_ij_masked(
        gray_i=gray_i,
        image_j_in_i=image_j_in_i,
        mask=projected_acceptance,
        saturation=saturation,
    )

    if plot:
        _plot_real_synth_error(
            gray_i.squeeze(0).squeeze(0).detach().clone().cpu().numpy(),
            image_j_in_i.squeeze(0).squeeze(0).detach().clone().cpu().numpy(),
            error.squeeze(0).detach().clone().cpu().numpy()
            * error[projected_acceptance].shape[0],
        )

    return error


def error_ij_masked(
    gray_i: torch.Tensor, image_j_in_i: torch.Tensor, mask: torch.Tensor, saturation=50,
) -> torch.Tensor:
    """Function to estimate the masked error

    Args:
        gray_i (torch.Tensor): image
        image_j_in_i (torch.Tensor): coordinates of points in image j represented in image i
        mask (torch.Tensor): points taken into consideration

    Returns:
        torch.Tensor: final error [HxW]
    """
    error = gray_i.new_zeros(image_j_in_i.size()).double()
    error[mask] = image_j_in_i[mask].double() - gray_i[mask].double()
    error = torch.clamp(error, -saturation, saturation)
    error = error / mask.int().sum(dim=2).sum(dim=2).unsqueeze(2).unsqueeze(2)
    error[torch.isnan(error)] = 0

    return error
