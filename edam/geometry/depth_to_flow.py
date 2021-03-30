import kornia
import torch


def _backproject_points(
    depth_i: torch.Tensor, k_i: torch.Tensor, normalize_points: bool,
) -> torch.Tensor:
    r"""Backproject points to 3d given a camera matrix and a depth image.

    Arguments:
        depth_i {torch.Tensor} -- Depth image i. [Nx1xHixWi]
        k_i {torch.Tensor} -- Camera matrix (not normalized) containing the camera
            intrinsics. [Nx3x3]
        normalize_points {bool} -- whether to normalise the pointcloud. This
            must be set to `True` when the depth is represented as the Euclidean
            ray length from the camera position.

    Returns:
        torch.Tensor -- 3D points in the same shape as the image [NxHxWx3].
    """
    # -- Backproject i points and tranform to j-camera coordinate system.

    i_points_3d_i: torch.Tensor = kornia.depth_to_3d(
        depth_i, k_i, normalize_points=normalize_points
    )  # Nx3xHxW
    i_points_3d_i = i_points_3d_i.permute(0, 2, 3, 1)  # NxHxWx3
    return i_points_3d_i  # NxHxWx3


def _transform_points(
    points_3d_i: torch.Tensor, i_pose_w: torch.Tensor, j_pose_w: torch.Tensor,
) -> torch.Tensor:
    r"""Transform points in i_coordinates to j_coordinates.

    Arguments:
        points_3d_i {torch.Tensor} -- Points in the i-coordinate system.
            Shape [NxHxWx3]
        i_pose_w {torch.Tensor} -- Transformation from world to camera i.
            Shape: [Bx4x4]
        j_pose_w {torch.Tensor} -- Transformation from world to camera j.
            Shape: [Bx4x4]

    Returns:
        torch.Tensor -- Points in the coordinates of the j camera. Mantains the shape
            [NxHxWx3]
    """

    j_trans_i = j_pose_w @ kornia.inverse_transformation(i_pose_w)

    points_3d_j = kornia.transform_points(
        j_trans_i[:, None], points_3d_i
    )  # NxHxWx3 -- points from i-image in j-coordinates
    return points_3d_j  # NxHxWx3
