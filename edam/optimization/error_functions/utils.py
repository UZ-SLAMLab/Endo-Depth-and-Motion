import matplotlib.pyplot as plt
import numpy as np
import torch


def skewmat(x_vec):
    """Create skew matrix from vector

    Args:
        x_vec ([type]): [description]

    Returns:
        [type]: [description]
    """
    W_row0 = x_vec.new_tensor([[0, 0, 0], [0, 0, 1], [0, -1, 0]]).view(3, 3)
    W_row1 = x_vec.new_tensor([[0, 0, -1], [0, 0, 0], [1, 0, 0]]).view(3, 3)
    W_row2 = x_vec.new_tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]).view(3, 3)

    x_skewmat = torch.stack(
        [
            torch.matmul(x_vec, W_row0.t()),
            torch.matmul(x_vec, W_row1.t()),
            torch.matmul(x_vec, W_row2.t()),
        ],
        dim=-1,
    )
    return x_skewmat


def d_proj_x(x_vec: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Derivative of the projection

    Args:
        x_vec (torch.Tensor): HxWx3
        k (torch.Tensor):calibration matrix

    Returns:
        torch.tensor: [description]
    """

    jac_d_proj_x = x_vec.new_zeros((x_vec.shape[0], x_vec.shape[1], 2, 3))
    jac_d_proj_x[:, :, 0, 0] = k[0, 0] / x_vec[:, :, 2]
    jac_d_proj_x[:, :, 1, 1] = k[1, 1] / x_vec[:, :, 2]
    jac_d_proj_x[:, :, 0, 2] = -k[0, 0] * x_vec[:, :, 0] / x_vec[:, :, 2] ** 2
    jac_d_proj_x[:, :, 1, 2] = -k[1, 1] * x_vec[:, :, 1] / x_vec[:, :, 2] ** 2
    return jac_d_proj_x


def _plot_real_synth_error(
    gray: np.array, synthetic_image: np.array, error_o: np.array
):
    """
    Plot function for real, synthetic image and the error.

    Just helper for debugging. Final visualization in main loop.

    gray: real image to synthetize.
    synthetic_image: synthetized image
    error_o: error between real and synthetize image in the keyframe.
    """
    fig = plt.figure()
    fig.add_subplot(1, 3, 1)
    plt.imshow(gray, vmin=0, vmax=255)
    fig.add_subplot(1, 3, 2)
    plt.imshow(synthetic_image, vmin=0, vmax=255)
    fig.add_subplot(1, 3, 3)
    plt.imshow(error_o, vmin=-70, vmax=70)
    plt.colorbar()
    plt.show()
