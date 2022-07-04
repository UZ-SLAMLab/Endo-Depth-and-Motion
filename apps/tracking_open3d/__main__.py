import argparse
from pathlib import Path
import pickle
import time
from typing import Dict, List
import os
import sys

from PIL import Image
import cv2 as cv
import kornia
from logzero import logger
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from edam.dataset import hamlyn
from edam.optimization.frame import create_frame
from edam.utils.file import list_files
from edam.utils.image.convertions import (
    numpy_array_to_pilimage,
    pilimage_to_numpy_array,
)
from edam.utils.image.pilimage import (
    pilimage_h_concat,
    pilimage_rgb_to_bgr,
    pilimage_v_concat,
)
from edam.utils.depth import depth_to_color
from edam.utils.LineMesh import LineMesh


def parse_args() -> argparse.Namespace:
    """Returns the ArgumentParser of this app.

    Returns:
        argparse.Namespace -- Arguments
    """
    parser = argparse.ArgumentParser(
        description="Shows the images from Hamlyn Dataset"
    )
    parser.add_argument(
        "-i",
        "--input-root-directory",
        type=str,
        required=True,
        help="Root directory where scans are find. E.G. path/to/hamlyn_tracking_test_data",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Device in where to run the optimization E.G. cpu, cuda:0, cuda:1",
    )
    parser.add_argument(
        "-fr",
        "--ratio_frame_keyframe",
        type=int,
        default=1,
        help="Number of frames until a new keyframes.",
    )
    parser.add_argument(
        "-o",
        "--folder_output",
        type=str,
        default="results",
        help="Folder where odometries are saved.",
    )
    parser.add_argument(
        "-t",
        "--tracking_type",
        type=str,
        default="point-to-plane",
        choices=["point-to-point", "point-to-plane", "park", "steinbrucker"],
        help="Select the type of ICP: point-to-point or point-to-plane.",
    )
    parser.add_argument(
        "-r",
        "--ransac",
        help="Execute a global registration with RANSAC to compute a pre-translation between the two point clouds "
             "before calculating the final translation with the local registration",
        action="store_true"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device_name = args.device
    list_scenes = list_files(args.input_root_directory)
    done = False
    scene_number = 0
    frame_number = 0
    scene_info = None
    vis_depth = o3d.visualization.Visualizer()
    vis_3d = o3d.visualization.Visualizer()
    vis_3d.create_window("HAMLYN3D", 1920 // 2, 1080 // 2)
    vis_depth.create_window("Depth", 256, 192, visible=False)
    estimated_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.0075)
    estimated_pose = np.eye(4)
    automode = False

    poses_register = dict()
    poses_register["scene_info"] = []
    poses_register["frame_number"] = []
    poses_register["estimated_pose"] = []
    folder_to_save_results = Path(args.folder_output)
    if not (folder_to_save_results.exists()):
        folder_to_save_results.mkdir(parents=True, exist_ok=True)
    output_path = folder_to_save_results / (time.strftime("%Y%m%d-%H%M%S") + ".pkl")

    while not done:
        begin = time.time()

        # -- Load info for the scene if it has not been loaded yet.
        if scene_info is None:
            logger.info("Reseting scene.")
            scene = list_scenes[scene_number]
            scene_info = hamlyn.load_scene_files(scene)
            frame_number = 0
            points = [np.array([0, 0, 0])]
            vis_3d.clear_geometries()
            vis_depth.clear_geometries()
            vis_3d.add_geometry(estimated_frame)

        # Get current camera
        ctr = vis_3d.get_view_control()
        cam = ctr.convert_to_pinhole_camera_parameters()

        # -- Open pose
        estimated_frame.transform(estimated_pose)

        # -- Open data
        (
            rgb_image_registered,
            depth_np,
            k_depth,
            h_d,
            w_d,
        ) = load_hamlyn_frame(scene_info, frame_number)

        depth_image = numpy_array_to_pilimage(
            (
                depth_to_color(
                    depth_np,
                    cmap="jet",
                    max_depth=np.max(depth_np),
                    min_depth=np.min(depth_np),
                )
            ).astype(np.uint8)
        )

        mask = depth_np > 0.3  # meters
        depth_np[mask] = 0.3

        gray = cv.cvtColor(
            pilimage_to_numpy_array(rgb_image_registered), cv.COLOR_BGR2GRAY
        )

        new_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(
                cv.cvtColor(
                    pilimage_to_numpy_array(rgb_image_registered), cv.COLOR_BGR2RGB
                )
            ),
            o3d.geometry.Image(depth_np),
            convert_rgb_to_intensity=False,
            depth_scale=1.0,
            depth_trunc=30000.0
        )

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            w_d,
            h_d,
            k_depth[0, 0, 0],  # fx
            k_depth[0, 1, 1],  # fy
            k_depth[0, 0, 2],  # cx
            k_depth[0, 1, 2],  # cy
        )

        new_frame = create_frame(
            c_pose_w=estimated_pose,
            c_pose_w_gt=None,
            gray_image=gray,
            rgbimage=None,
            depth=depth_np,
            k=k_depth.numpy().reshape(3, 3),
            idx=frame_number,
            ref_camera=(frame_number == 0),
            scales=0,
            code_size=128,
            device_name=device_name,
            uncertainty=None,
        )

        # Keyframe updating
        if frame_number == 0:
            relative_estimated_pose = np.eye(4)
            ref_estimated_pose = np.eye(4)
            ref_rgb_image_registered = rgb_image_registered.copy()
            ref_depth_image = depth_image.copy()
            cam.extrinsic = np.eye(4)

        estimated_pose = ref_estimated_pose @ relative_estimated_pose
        new_frame.modify_pose(c_pose_w=estimated_pose)

        new_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            new_rgbd_image, intrinsic, new_frame.c_pose_w,
        )

        if frame_number != 0:
            threshold = 0.03
            trans_init = np.eye(4)

            print("Initial alignment")
            evaluation = o3d.pipelines.registration.evaluate_registration(
                ref_pcd,
                new_pcd,
                threshold,
                trans_init,
            )
            print(evaluation)

            if args.ransac:
                down_ref_pcd, fpfh_ref_pcd = preprocess_point_cloud(ref_pcd, voxel_size=0.001)
                down_new_pcd, fpfh_new_pcd = preprocess_point_cloud(new_pcd, voxel_size=0.001)
                result_ransac = execute_global_registration(down_ref_pcd, down_new_pcd,
                                                            fpfh_ref_pcd, fpfh_new_pcd,
                                                            voxel_size=0.001)
                trans_init = result_ransac.transformation

                print("Alignment after global registration")
                evaluation = o3d.pipelines.registration.evaluate_registration(
                    ref_pcd,
                    new_pcd,
                    threshold,
                    trans_init,
                )
                print(evaluation)

            if args.tracking_type == "point-to-point":
                print("Apply point-to-point ICP")
                # Relative estimated pose between two adjacent pcd
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    ref_pcd,
                    new_pcd,
                    threshold,
                    trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
                )
                print(reg_p2p)
                relative_estimated_pose = reg_p2p.transformation

            elif args.tracking_type == "point-to-plane":
                print("Apply point-to-plane ICP")
                if not args.ransac:
                    # Voxel downsampling and feature extraction
                    down_ref_pcd, fpfh_ref_pcd = preprocess_point_cloud(ref_pcd, voxel_size=0.001)
                    down_new_pcd, fpfh_new_pcd = preprocess_point_cloud(new_pcd, voxel_size=0.001)

                # Voxel downsampling and normal estimation
                down_ref_pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.002, max_nn=15))
                down_new_pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.002, max_nn=15))

                # Relative estimated pose between two adjacent pcd
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    down_ref_pcd,
                    down_new_pcd,
                    threshold,
                    trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
                )
                print(reg_p2p)
                relative_estimated_pose = reg_p2p.transformation

            elif args.tracking_type == "steinbrucker":
                option = o3d.pipelines.odometry.OdometryOption()
                [_, relative_estimated_pose, _] = o3d.pipelines.odometry.compute_rgbd_odometry(
                    new_rgbd_image, ref_rgbd_image, intrinsic, trans_init,
                    o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(), option
                )
                relative_estimated_pose = np.linalg.inv(relative_estimated_pose)

            elif args.tracking_type == "park":
                option = o3d.pipelines.odometry.OdometryOption()
                [_, relative_estimated_pose, _] = o3d.pipelines.odometry.compute_rgbd_odometry(
                    new_rgbd_image, ref_rgbd_image, intrinsic, trans_init,
                    o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option
                )
                relative_estimated_pose = np.linalg.inv(relative_estimated_pose)

        # Update keyframe
        if frame_number % args.ratio_frame_keyframe == 0:
            ref_estimated_pose = ref_estimated_pose @ relative_estimated_pose
            ref_rgb_image_registered = rgb_image_registered
            ref_depth_image = depth_image
            ref_rgbd_image = new_rgbd_image
            ref_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                new_rgbd_image, intrinsic, ref_estimated_pose,
            )
            vis_3d.add_geometry(new_pcd)
            logger.info("KEYFRAME INSERTED")

        end = time.time()
        time_one_image = round((end - begin) * 1000)
        print("   Computing time {}ms".format(time_one_image))

        frame = pilimage_v_concat(
            [
                pilimage_h_concat(
                    [ref_rgb_image_registered.resize((round(1920/4), round(h_d*(1920/4)/w_d)), resample=Image.BILINEAR),
                     ref_depth_image.resize((round(1920 / 4), round(h_d * (1920 / 4) / w_d)), resample=Image.BILINEAR),
                     rgb_image_registered.resize((round(1920/4), round(h_d*(1920/4)/w_d)), resample=Image.BILINEAR),
                     depth_image.resize((round(1920 / 4), round(h_d * (1920 / 4) / w_d)), resample=Image.BILINEAR)
                     ]
                ),
            ]
        )

        estimated_frame.transform(np.linalg.inv(estimated_pose))
        vis_3d.update_geometry(estimated_frame)

        if frame_number > 0:
            translation = np.linalg.inv(estimated_pose)[:, 3]
            new_point = translation[:3]
            points.append(new_point)
            line_mesh = LineMesh(points, radius=0.0005)
            line_mesh_geoms = line_mesh.cylinder_segments
            vis_3d.add_geometry(*line_mesh_geoms)
            del points[0]

        logger.info(
            f"frame {frame_number:d}"
        )
        logger.info(f"")

        poses_register["scene_info"].append(scene_info)
        poses_register["frame_number"].append(frame_number)
        poses_register["estimated_pose"].append(new_frame.c_pose_w.copy())

        a_file = open(str(output_path), "wb")
        pickle.dump(poses_register, a_file)
        a_file.close()

        camera_ = o3d.camera.PinholeCameraParameters()
        camera_.intrinsic = cam.intrinsic
        camera_.extrinsic = cam.extrinsic
        ctr.convert_from_pinhole_camera_parameters(camera_)
        cv.imshow("HAMLYN", pilimage_to_numpy_array(frame))

        while cv.getWindowProperty("HAMLYN", 0) >= 0:

            vis_3d.poll_events()
            vis_3d.update_renderer()
            key = cv.waitKey(1) & 0xFF
            if automode:
                if key == ord("s"):
                    automode = False
                frame_number = (frame_number + 1) % len(scene_info["list_color_images"])
                break
            if key == ord("a"):
                automode = True
                logger.info(help_functions()[chr(key)])
                frame_number = (frame_number + 1) % len(scene_info["list_color_images"])
                break
            if key == ord("n"):
                logger.info(help_functions()[chr(key)])
                frame_number = (frame_number + 1) % len(scene_info["list_color_images"])
                break
            if key == ord("j"):
                logger.info(help_functions()[chr(key)])
                frame_number = (frame_number + 10) % len(
                    scene_info["list_color_images"]
                )
                break
            elif key == ord("p"):
                logger.info(help_functions()[chr(key)])
                frame_number = frame_number - 1
                if frame_number < 0:
                    frame_number = len(scene_info["list_color_images"]) - 1
                break
            elif key == ord(" "):
                logger.info(help_functions()[chr(key)])
                scene_number = (scene_number + 1) % len(list_scenes)
                frame_number = 0
                scene_info = None
                break
            elif key == ord("\x08"):
                logger.info(help_functions()[chr(key)])
                scene_number = scene_number - 1
                if scene_number < 0:
                    scene_number = len(list_scenes) - 1
                frame_number = 0
                scene_info = None
                break
            elif key == ord("h"):
                logger.info(help_functions()[chr(key)])
                print_help()
            elif key == ord("q"):
                logger.info(help_functions()[chr(key)])
                done = True
                vis_3d.destroy_window()
                vis_depth.destroy_window()
                break
            elif key != 255:
                logger.info(f"Unkown command, {chr(key)}")
                print("Use 'h' to print help and 'q' to quit.")


def load_hamlyn_frame(scene_info: Dict[str, List[str]], frame_number: int):
    """Function to load the scene and frame from the Hamlyn dataset,

    Args:
        scene_info ([type]): [description]
        frame_number (int): [description]

    Returns:
        [type]: [description]
    """
    # -- Open the rgb image.
    rgb_path = scene_info["list_color_images"][frame_number]
    rgb_image = pilimage_rgb_to_bgr(Image.open(rgb_path))

    # -- Open the depth image.
    depth_path = scene_info["list_depth_images"][frame_number]
    depth_np = cv.imread(depth_path, cv.IMREAD_ANYDEPTH).astype(np.float32) / 1000

    # -- Get the camera matrices.
    k_depth: np.ndarray = scene_info["intrinsics"][:3, :3]  # type: ignore

    # -- Transform into tensors.
    k_depth = kornia.utils.image_to_tensor(k_depth, keepdim=False).squeeze(1)  # Bx3x3

    h_d, w_d = depth_np.shape

    return (
        rgb_image,
        depth_np,
        k_depth,
        h_d,
        w_d,
    )


def preprocess_point_cloud(pcd, voxel_size):
    # Downsample with a voxel size
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # Estimate normal with search radius
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # Compute FPFH feature with search radius
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def help_functions():
    helper = {}
    helper["a"] = "Autoplay"
    helper["s"] = "Stop Autoplay"
    helper["n"] = "Next Image"
    helper["j"] = "Next Image x10"
    helper["p"] = "Previous Image"
    helper["\x08"] = "(Backslash) Previous Scene"
    helper[" "] = "(Space bar) Next Scene"
    helper["h"] = "Print help"
    helper["q"] = "Quit program"
    return helper


def print_help():
    logger.info("Printing help:")
    for k, v in help_functions().items():
        print(k, ":\t", v)


if __name__ == "__main__":
    main()
