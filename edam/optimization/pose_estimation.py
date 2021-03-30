import time

from logzero import logger
from pytorch3d.transforms.so3 import so3_exponential_map
import torch
import torch.autograd

from edam.optimization.error_functions.pose_odometry import (
    error_pose_optimization,
    error_pose_optimization_jac,
)
from edam.optimization.error_functions.rotation_odometry import (
    error_pose_rotation_jac,
    error_rotation_optimization,
)
from edam.optimization.frame import Frame
from edam.optimization.optimization_algorithms import gauss_newton


class PoseEstimation:
    """
    Class to perform pose estimation given a reference keyframe
    """

    def __init__(self, preoptimize_rotation: bool = True) -> None:
        """
        Constructor for the pose estimation
        """
        self.preoptimize_rotation = preoptimize_rotation

    def set_ref_keyframe(self, Keyframe: Frame) -> None:
        """Set a reference keyframe to track

        Args:
            Keyframe (Frame): Keyframe with depth to track.
        """
        self.reference_keyframe = Keyframe

    def frames_to_numpy_array(self, frame_to_track: Frame) -> torch.Tensor:
        """
        Create a paramater array for the SE(3) optimization

        Returns:
            np.array: parameter array
        """
        # All the elements initializated as zero
        return torch.zeros((6,)).to(frame_to_track.device)

    def frames_to_numpy_array_rot(self, frame_to_track: Frame) -> torch.Tensor:
        """
        Create a paramater array for the rotation optimization

        Returns:
            np.array: parameter array
        """
        # All the elements initializated as zero
        return torch.zeros((3,)).to(frame_to_track.device)

    def update_frames(self, x: torch.Tensor, frame_to_track: Frame) -> None:
        """
        Updating frames after the optimization

        Args:
            frames_to_process (List[Frame]): [description]

        Returns:
            np.array: [description]
        """
        # Updating pose
        Rupd_ = so3_exponential_map(x[0:3].unsqueeze(0)).double()
        tupd_ = x[3:6]
        frame_to_track.c_pose_w_tc[0:3, 0:3] = (
            Rupd_ @ frame_to_track.c_pose_w_tc[0:3, 0:3]
        )
        frame_to_track.c_pose_w_tc[0:3, 3] = (
            Rupd_ @ frame_to_track.c_pose_w_tc[0:3, 3] + tupd_
        )
        frame_to_track.c_pose_w = frame_to_track.c_pose_w_tc.detach().cpu().numpy()

    def update_frames_rot(self, x: torch.Tensor, frame_to_track: Frame) -> None:
        """
        Updating frames after the optimization

        Args:
            frames_to_process (List[Frame]): [description]

        Returns:
            np.array: [description]
        """
        idx = 0

        # Updating pose
        Rupd_ = so3_exponential_map(x[idx : idx + 3].unsqueeze(0)).double()
        frame_to_track.c_pose_w_tc[0:3, 0:3] = (
            Rupd_ @ frame_to_track.c_pose_w_tc[0:3, 0:3]
        )
        frame_to_track.c_pose_w_tc[0:3, 3] = Rupd_ @ frame_to_track.c_pose_w_tc[0:3, 3]
        frame_to_track.c_pose_w = frame_to_track.c_pose_w_tc.detach().cpu().numpy()

    def run(
        self, frame_to_track: Frame, verbose: bool = True, show_error: bool = False
    ) -> None:
        """
        Evaluate the photometric cost of the problem with the current state.

        Returns:
            float: photometric error accumulated
        """
        a = time.time()
        max_ite = 150
        step = 10e-5

        for scale in range(frame_to_track.number_of_pyr - 1, -1, -1):

            if (scale == frame_to_track.number_of_pyr - 1) and (
                self.preoptimize_rotation
            ):

                def initial_frame_to_np_():
                    return {"x": self.frames_to_numpy_array_rot(frame_to_track)}

                def fun_to_optimize_(x_dict):
                    x = x_dict["x"]
                    return error_rotation_optimization(
                        x,
                        frame_to_track,
                        self.reference_keyframe,
                        scale=scale,
                        plot=False,
                    )

                def jac_func_(x_dict, error_=None):
                    x = x_dict["x"]
                    return error_pose_rotation_jac(
                        x,
                        frame_to_track,
                        self.reference_keyframe,
                        scale=scale,
                        plot=False,
                    )

                def update_frames_(x_dict):
                    x = x_dict["x"]
                    self.update_frames_rot(x, frame_to_track)

                gauss_newton(
                    fun_to_optimize_,
                    initial_frame_to_np_,
                    update_frames_,
                    estimate_jacobian=jac_func_,
                    max_ite=max_ite,
                    verbose=verbose,
                )

            def initial_frame_to_np():
                return {"x": self.frames_to_numpy_array(frame_to_track)}

            def fun_to_optimize(x_dict):
                x = x_dict["x"]
                return error_pose_optimization(
                    x, frame_to_track, self.reference_keyframe, scale=scale, plot=False,
                )

            def jac_func(x_dict, error_=None):
                x = x_dict["x"]
                return error_pose_optimization_jac(
                    x, frame_to_track, self.reference_keyframe, scale=scale, plot=False,
                )

            def update_frames(x_dict):
                x = x_dict["x"]
                self.update_frames(
                    x, frame_to_track,
                )

            gauss_newton(
                fun_to_optimize,
                initial_frame_to_np,
                update_frames,
                estimate_jacobian=jac_func,
                max_ite=max_ite,
                verbose=verbose,
            )

        b = time.time()

        logger.info(str((b - a) * 1000) + " ms for pose estimation")
