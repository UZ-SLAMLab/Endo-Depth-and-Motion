import kornia
import numpy as np
from pytorch3d.transforms.so3 import so3_exponential_map
import torch

from edam.geometry.depth_to_flow import _backproject_points, _transform_points
from edam.optimization.frame import Frame

from .photometric_error import photometric_error_gen
from .utils import d_proj_x, skewmat


def error_rotation_optimization(
    x: torch.Tensor,
    frame_to_track: Frame,
    ref_frame: Frame,
    scale: int = 0,
    plot: bool = False,
):
    """
    Estimate the photometric error of the back-projection of the
    first frame observed in a second frame. It only considers the
    rotation.

    Args:
        x (torch.Tensor) : state vector [yaw pich roll]_i'
        ref_frame (Frame): reference frame
        frame_to_track (Frame): observer frame
        scale(int): level in the multiscale
        plot(bool): plot the error
    """
    # Updating pose
    Rupd_ = so3_exponential_map(x[0:3].unsqueeze(0)).double()
    j_pose_w = frame_to_track.c_pose_w_tc.clone().detach().unsqueeze(0).double()
    j_pose_w[:, 0:3, 0:3] = Rupd_ @ j_pose_w[:, 0:3, 0:3]
    j_pose_w[0, 0:3, 3] = Rupd_ @ j_pose_w[0, 0:3, 3]
    margins = 12 / (2 ** scale)
    error = photometric_error_gen(
        i_pose_w=ref_frame.c_pose_w_tc.unsqueeze(0),
        depth_ref=ref_frame.depth,
        k_ref=ref_frame.k_tc.unsqueeze(0),
        j_pose_w=j_pose_w,
        k_j_target=frame_to_track.pyr_k_tc[scale].unsqueeze(0),
        gray_i=ref_frame.gray_tc.unsqueeze(0),
        gray_j=frame_to_track.pyr_gray_tc[scale].unsqueeze(0).unsqueeze(0),
        margins=margins,
        saturation=120,
        plot=plot,
    )

    if ref_frame.pseudo_uncertainty_map is not None:
        error = ref_frame.pseudo_uncertainty_map * error

    return error.flatten()


def error_pose_rotation_jac(
    x: torch.Tensor,
    frame_to_track: Frame,
    ref_frame: Frame,
    scale: int = 0,
    plot: bool = False,
    gt: bool = False,
):
    """
    Estimate the photometric error of the back-projection of the
    first frame observed in a second frame. This function takes
    as parameters the Lie algebra of the rotation and translation
    of the camera optimized.

    Args:
    x (torch.Tensor) : state vector [x y z yaw pich roll]_i
    Frame_ref (Frame): reference frame
    Frame_next (Frame): observer frame
    """
    # Updating pose
    Rupd_ = so3_exponential_map(x[0:3].unsqueeze(0)).double()
    j_pose_w = frame_to_track.c_pose_w_tc.clone().detach().unsqueeze(0).double()
    j_pose_w[:, 0:3, 0:3] = Rupd_ @ j_pose_w[:, 0:3, 0:3]
    j_pose_w[0, 0:3, 3] = Rupd_ @ j_pose_w[0, 0:3, 3]
    if gt:
        j_pose_w = frame_to_track.c_pose_w_gt_tc.unsqueeze(0).double()

    margins = 12 / (2 ** scale)

    jac = photometric_error_jac_rotation(
        i_pose_w=ref_frame.c_pose_w_tc.unsqueeze(0),
        depth_ref=ref_frame.depth,
        k_ref=ref_frame.k_tc.unsqueeze(0),
        j_pose_w=j_pose_w,
        k_j_target=frame_to_track.pyr_k_tc[scale].unsqueeze(0),
        gray_i=ref_frame.gray_tc.unsqueeze(0),
        gray_j=frame_to_track.pyr_gray_tc[scale].unsqueeze(0).unsqueeze(0),
        saturation=120,
        margins=margins,
        uncertainty=ref_frame.pseudo_uncertainty_map,
    )

    return jac


def photometric_error_jac_rotation(
    i_pose_w: torch.Tensor,
    depth_ref: torch.Tensor,
    k_ref: torch.Tensor,
    j_pose_w: torch.Tensor,
    k_j_target: torch.Tensor,
    gray_i: torch.Tensor,
    gray_j: torch.Tensor,
    uncertainty: torch.Tensor = None,
    margins: int = 12,
    saturation: int = 254,
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
    jac_image = kornia.spatial_gradient(gray_j, mode="diff")
    jac_xy = (
        torch.nn.functional.grid_sample(
            jac_image.squeeze(0).double(), i_points_2d_j_norm, align_corners=True,  # type: ignore
        )
        .permute(0, 2, 3, 1)
        .squeeze(0)
        .unsqueeze(2)
    ).double()

    jac_d_proj_x = d_proj_x(i_points_3d_j.squeeze(0), k_j_target.squeeze(0)).double()
    jac_se3 = jac_d_proj_x.new_zeros((gray_i.shape[-2], gray_i.shape[-1], 3, 3))
    jac_se3[:, :, :, 0:3] = -(
        skewmat(i_points_3d_j.reshape(1, -1, 3).double())
        .reshape(1, gray_i.shape[-2], gray_i.shape[-1], 3, 3)
        .double()
    ).squeeze(0)

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

    jac = jac_xy @ jac_d_proj_x @ jac_se3
    if uncertainty is not None:
        jac = uncertainty * jac
    jac = jac.reshape(-1, 3) / projected_acceptance.int().sum(dim=2).sum(dim=2)

    return jac.squeeze(1)
