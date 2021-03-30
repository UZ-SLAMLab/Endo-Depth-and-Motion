from typing import Callable, Optional

import cv2 as cv
import numpy as np

from logzero import logger
import torch


class Frame:
    """
    Frame class
    """

    def __init__(
        self,
        c_pose_w: np.array,
        c_pose_w_gt: Optional[np.array],
        gray: np.array,
        rgbimage: Optional[torch.Tensor],
        depth: Optional[np.array],
        k: np.array,
        idx: int,
        ref_camera: bool = False,
        number_of_pyr: int = 3,
        code_size: int = 128,
        device_name: str = "cpu",
        uncertainty: torch.Tensor = None,
    ) -> None:
        """
        Initialization of the frame class. This camera class will be optimized in the bundle adjustment
        or the pose optimization. It is composed of a position Tcw in the world, the gray image
        observed and the inverse depth. 

        Args:
            c_pose_w (np.array): [description]
            c_pose_w_gt (np.array): Tcw
            gray (np.array): gray image
            depth (np.array): depth image estimated. Only used if keyframe
            k (np.array): calibration matrix
            idx (int): index of the frame
            ref_camera (bool, optional): If true it becomes keyframe. Defaults to False.
            number_of_pyr (int, optional): number of pyramids used in the optimization. Defaults to 3.
            device_name: device to perform the optimization # Default coming from cpu
        """
        self.h, self.w = gray.shape[0], gray.shape[1]
        device = torch.device(device_name)  # device to perform the optimization
        self.device = device

        self.pyr_k = []  # pyramids of k
        self.pyr_k_tc = []  # Pyramids of k in torch

        self.pyr_gray = []  # pyramid of grays
        self.pyr_gray_tc = []  # pyramid of grays in torch
        pyr_normal_gray_ = []  # Pyramids of grays to generate

        self.k = np.copy(k)
        self.k_tc = torch.from_numpy(self.k).float().to(device)
        self.pyr_k.append(self.k.copy())
        self.pyr_k_tc.append(self.k_tc.clone().detach())

        self.pyr_gray.append(gray)
        pyr_normal_gray_.append(gray)
        self.pyr_gray_tc.append(torch.from_numpy(self.pyr_gray[-1]).float().to(device))
        self.gray = np.copy(gray)
        self.gray_tc = torch.from_numpy(self.gray).unsqueeze(0).float().to(device)
        self.number_of_pyr = number_of_pyr

        # Create scales
        for i in range(0, number_of_pyr):
            pyr_normal_gray_.append(cv.pyrDown(pyr_normal_gray_[-1]))
            self.pyr_gray.append(pyr_normal_gray_[-1].copy())
            self.pyr_k.append(self.pyr_k[-1].copy())
            self.pyr_k[-1][0:2, 0:3] /= 2
            self.pyr_k_tc.append(self.pyr_k_tc[-1].clone().detach())
            self.pyr_k_tc[-1][0:2, 0:3] /= 2
            self.pyr_gray_tc.append(
                torch.from_numpy(self.pyr_gray[-1]).float().to(device)
            )

        # Poses initialization
        self.c_pose_w_tc = torch.from_numpy(c_pose_w).double().to(device)
        self.c_pose_w = np.copy(c_pose_w)

        # Ground truth poses initialization
        self.c_pose_w_gt = None
        self.c_pose_w_gt_tc = None

        if c_pose_w_gt is not None:
            self.c_pose_w_gt = np.copy(c_pose_w_gt)
            self.c_pose_w_gt_tc = torch.from_numpy(self.c_pose_w_gt).double().to(device)

        # Depth given in the initialization
        if depth is not None:
            self.depth = (
                (torch.from_numpy(depth).float().to(device)).unsqueeze(0).unsqueeze(0)
            )  # Initial estimation of the inverse depth
        self.code = torch.zeros((1, code_size), requires_grad=True,).to(device)  #

        self.pseudo_uncertainty_map = None

        self.rgbimage = rgbimage

        if uncertainty is not None:
            self.pseudo_uncertainty_map = uncertainty.clone()

        self.ref_camera = ref_camera

        self.idx = idx

    def modify_pose(self, c_pose_w):
        """Function to modify the depth
        both from numpy variable and torch variable. 

        Args:
            c_pose_w (np.array): Pose T_cw
        """
        self.c_pose_w = c_pose_w  # cpu
        self.c_pose_w_tc = torch.from_numpy(c_pose_w).double().to(self.device)

    def modify_code(self, code):
        self.code = code


def create_frame(
    c_pose_w: np.array,
    c_pose_w_gt: Optional[np.array],
    gray_image: np.array,
    rgbimage: Optional[torch.Tensor],
    depth: Optional[np.array],
    k: np.array,
    idx: int,
    ref_camera: bool,
    scales: int,
    code_size: int,
    device_name: str = "cpu",
    uncertainty: torch.Tensor = None,
) -> Frame:
    """
    Function to create frames
    Args:
            c_pose_w (np.array): [description]
            c_pose_w_gt (np.array): Tcw
            gray (np.array): gray image
            depth (np.array): depth image estimated. Only used if keyframe
            model(np)
            k (np.array): calibration matrix
            idx (int): index of the frame
            ref_camera (bool, optional): If true it becomes keyframe. Defaults to False.
            number_of_pyr (int, optional): number of pyramids used in the optimization. 
                    Defaults to 3.
            device_name: device to perform the optimization
    Returns:
            new frame
    """
    return Frame(
        c_pose_w=c_pose_w,
        c_pose_w_gt=c_pose_w_gt,
        gray=gray_image,
        rgbimage=rgbimage,
        depth=depth,
        k=k,
        idx=idx,
        number_of_pyr=scales,
        ref_camera=ref_camera,
        code_size=code_size,
        device_name=device_name,
        uncertainty=uncertainty,
    )
