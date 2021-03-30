import kornia
import torch

from edam.geometry.depth_to_flow import _backproject_points, _transform_points


def synthetise_image_and_error(
    i_pose_w: torch.Tensor,
    depth_ref: torch.Tensor,
    k_ref: torch.Tensor,
    j_pose_w: torch.Tensor,
    k_target: torch.Tensor,
    gray_ref: torch.Tensor,
    gray_target: torch.Tensor,
):
    """
    Estimate the photometric error of the back-projection of the
    first frame observed in a second frame.


    Args:
        i_pose_w: torch.tensor, # [1x4x4]
        depth_ref: torch.tensor, # [1xHxW]
        K_ref: torch.tensor, # [1x3x3]
        j_pose_w: torch.tensor, # [Bx4x4]
        K_target: torch.tensor, # [Bx3x3]
        gray_ref: torch.tensor, # [1xHxW]
        gray_target: torch.tensor, # [BxHxW]
        plot: bool, plot image. Just for debugging.

    Return: 
        error: error vector [(H,W)]
        synthetic_image:error vector [(H,W)]
    """
    i_points_3d_i = _backproject_points(
        depth_ref, k_ref.unsqueeze(0), normalize_points=False,
    )  # NxHxWx3

    i_points_3d_j = _transform_points(
        i_points_3d_i.float(),
        i_pose_w.unsqueeze(0).float(),
        j_pose_w.unsqueeze(0).float(),
    )  # NxHxWx3
    camera_matrix_tmp: torch.Tensor = k_target.reshape(
        1, 1, 1, k_target.shape[0], k_target.shape[1]
    )  # Nx1x1xHxW

    i_points_2d_j: torch.Tensor = kornia.project_points(
        i_points_3d_j, camera_matrix_tmp
    ).squeeze(0)

    height, width = gray_target.shape[-2:]
    i_points_2d_j_norm: torch.Tensor = kornia.normalize_pixel_coordinates(
        i_points_2d_j, height, width
    ).double()

    image_j_in_i = torch.nn.functional.grid_sample(
        gray_target.unsqueeze(0).unsqueeze(0).to(dtype=torch.float64),
        i_points_2d_j_norm.unsqueeze(0),
        align_corners=True,  # type: ignore
    )
    # Not take into account the black in the gray image of the first keyframe
    black_acceptance = gray_ref > 0

    # Hard code margins 12.
    width_acceptance = torch.bitwise_and(
        i_points_2d_j[..., 0] >= 12, i_points_2d_j[..., 0] < gray_target.shape[1] - 12,
    )
    height_acceptance = torch.bitwise_and(
        i_points_2d_j[..., 1] >= 12, i_points_2d_j[..., 1] < gray_target.shape[0] - 12,
    )
    projected_acceptance = torch.bitwise_and(
        black_acceptance, torch.bitwise_and(width_acceptance, height_acceptance)
    )

    error = gray_ref.new_zeros(gray_ref.shape, dtype=torch.double)
    error[projected_acceptance] = (
        image_j_in_i.squeeze(0)[projected_acceptance].double()
        - gray_ref[projected_acceptance].double()
    ) / error[projected_acceptance].shape[0]
    synthetic_image = gray_target.new_zeros(gray_target.shape, dtype=gray_target.dtype)
    i_points_2d_j_aux = i_points_2d_j[projected_acceptance.squeeze(0), :].round().long()
    synthetic_image[i_points_2d_j_aux[:, 1], i_points_2d_j_aux[:, 0]] = gray_ref[
        projected_acceptance
    ]

    return (
        error.squeeze(0).detach().cpu().numpy(),
        synthetic_image.detach().cpu().numpy(),
    )
