import cv2 as cv
import open3d as o3d
import pickle
import numpy as np
import kornia
from pathlib import Path
import argparse
import time
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from edam.utils.parser import txt_to_nparray
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
        "-o",
        "--output_root_directory",
        type=str,
        required=True,
        help="Root directory where mesh is saved. E.g. path/to/test1",
    )
    parser.add_argument(
        "-i",
        "--input_odometry_file",
        type=str,
        required=True,
        help="Input odometry file. E.g. apps/tracking_ours/results/test1.pkl",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    root = args.output_root_directory
    odometry_file = args.input_odometry_file

    name_of_folder_to_save_results = os.path.splitext(os.path.basename(os.path.normpath(odometry_file)))[0]
    parent_folder_to_save_results = Path(os.path.join(root, "map"))
    if not (parent_folder_to_save_results.exists()):
        parent_folder_to_save_results.mkdir(parents=True, exist_ok=True)
    folder_to_save_results = Path(os.path.join(parent_folder_to_save_results, name_of_folder_to_save_results))
    if not (folder_to_save_results.exists()):
        folder_to_save_results.mkdir(parents=True, exist_ok=True)

    with open(odometry_file, 'rb') as f:
        poses_register = pickle.load(f)
    poses = poses_register["estimated_pose"]
    frame_numbers = poses_register["frame_number"]

    # Create poses.log
    with open(os.path.join(folder_to_save_results, 'poses.log'), 'w') as traj:
        for i, pose in enumerate(poses):
            traj.write(f"0 0 {i}\n"
                       f"{pose[0, 0]} {pose[0, 1]} {pose[0, 2]} {pose[0, 3]}\n"
                       f"{pose[1, 0]} {pose[1, 1]} {pose[1, 2]} {pose[1, 3]}\n"
                       f"{pose[2, 0]} {pose[2, 1]} {pose[2, 2]} {pose[2, 3]}\n"
                       f"{pose[3, 0]} {pose[3, 1]} {pose[3, 2]} {pose[3, 3]}\n"
                       )
    trajectory = o3d.io.read_pinhole_camera_trajectory(os.path.join(folder_to_save_results, 'poses.log'))
    os.remove(os.path.join(folder_to_save_results, 'poses.log'))
    _, _, intrinsic = load_frame(root, i)
    for i in frame_numbers:
        print("Changing intrinsics of the {:d}-th image.".format(i))
        trajectory.parameters[i].intrinsic = intrinsic
    o3d.io.write_pinhole_camera_trajectory(os.path.join(folder_to_save_results, 'trajectory.log'), trajectory)

    rgbd_images = compute_rgbd_images(root, frame_numbers)

    # Volumetric fusion with TSDF
    begin = time.time()
    mesh = tsdf_optimize(root, frame_numbers, rgbd_images, trajectory)
    end = time.time()
    time_tsdf = round((end - begin) * 1000)
    print("   Computing time of volumetric fusion {}ms".format(time_tsdf))
    o3d.io.write_triangle_mesh(os.path.join(folder_to_save_results, 'mesh.ply'),
                               mesh,
                               write_ascii=True)

    # Draw trajectory
    points = []
    for i, pose in enumerate(poses):
        pose = np.linalg.inv(pose)
        position = pose[:, 3]
        points.append(position[:3])
    line_mesh = LineMesh(points, radius=0.0005)
    line_mesh_geoms = line_mesh.cylinder_segments

    o3d.visualization.draw_geometries(line_mesh_geoms + [mesh])

    print("\n-> Done!")


def compute_rgbd_images(root, frame_numbers):
    rgbd_images = []
    for i in frame_numbers:
        print("Store {:d}-th image into rgbd_images.".format(i))
        color, depth, _ = load_frame(root, i)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            o3d.geometry.Image(depth),
            depth_trunc=30000.0,
            depth_scale=1.0,
            convert_rgb_to_intensity=False
        )
        rgbd_images.append(rgbd)

    return rgbd_images


def tsdf_optimize(root, frame_numbers, rgbd_images, poses):
    tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.001,
        sdf_trunc=0.005,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    for i in frame_numbers:
        print("Integrate {:d}-th image into the volume.".format(i))
        _, _, intrinsic = load_frame(root, i)
        tsdf.integrate(rgbd_images[i], intrinsic, np.linalg.inv(poses.parameters[i].extrinsic))

    print("Extract a triangle mesh from the volume and visualize it.")
    mesh = tsdf.extract_triangle_mesh()

    return mesh


def load_frame(root, frame_number: int):
    color = o3d.io.read_image(os.path.join(root, "color", "{:010d}.jpg".format(frame_number)))
    depth = cv.imread(os.path.join(root, "depth", "{:010d}.png".format(frame_number)), cv.IMREAD_ANYDEPTH)\
                .astype(np.float32) / 1000  # meters
    intrinsics = txt_to_nparray(os.path.join(root, "intrinsics.txt"))
    k: np.ndarray = intrinsics[:3, :3]  # type: ignore

    # -- Transform into tensors.
    k = kornia.utils.image_to_tensor(k, keepdim=False).squeeze(1)  # Bx3x3

    h, w = depth.shape

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        w,
        h,
        k[0, 0, 0],  # fx
        k[0, 1, 1],  # fy
        k[0, 0, 2],  # cx
        k[0, 1, 2],  # cy
    )

    return color, depth, intrinsic


if __name__ == "__main__":
    main()
