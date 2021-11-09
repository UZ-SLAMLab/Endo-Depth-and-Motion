# Endo-Depth-and-Motion

This repository contains the code of *Endo-Depth*'s depth prediction from single images, the photometric and the others trackings methods and the volumetric fusion used in the paper

> **Localization and Reconstruction in Endoscopic Videos using Depth Networks and Photometric Constraints**
>
> [David Recasens](https://davidrecasens.github.io/), [José Lamarca](https://webdiis.unizar.es/~jlamarca/), [José M. Fácil](https://webdiis.unizar.es/~jmfacil/), [José María M. Montiel](https://janovas.unizar.es/sideral/CV/jose-maria-martinez-montiel) and [Javier Civera](https://janovas.unizar.es/sideral/CV/javier-civera-sancho)
>
> I3A, University of Zaragoza
> 
> [RA-L paper](https://ieeexplore.ieee.org/abstract/document/9478277/)
> [arXiv paper](https://arxiv.org/abs/2103.16525)
> [IROS 2021 video presentation](https://youtu.be/YfXkK9R0htE)

<p align="center">
  <img src="assets/teaser.gif" alt="example input output gif" width="600" />
</p>
<p align="center">
  <a href="https://youtu.be/G1XWIyEbvPc">Full video</a> of <i>Endo-Depth-and-Motion</i> working on Hamlyn dataset
</p>

```shell
@article{recasens2021endo,
  title={Endo-Depth-and-Motion: Reconstruction and Tracking in Endoscopic Videos Using Depth Networks and Photometric Constraints},
  author={Recasens, David and Lamarca, Jos{\'e} and F{\'a}cil, Jos{\'e} M and Montiel, JMM and Civera, Javier},
  journal={IEEE Robotics and Automation Letters},
  volume={6},
  number={4},
  pages={7225--7232},
  year={2021},
  publisher={IEEE}
}
```

## 💭 About

*Endo-Depth-and-Motion* is a pipeline where first, pixel-wise depth is predicted on a set of keyframes of the endoscopic monocular video using a deep neural network (*Endo-Depth*). The motion of each frame with respect to the closest keyframe is estimated by minimizing the photometric error, robustified using image pyramids and robust error functions. Finally, the depth maps of the keyframes are fused in a Truncated Signed Distance Function (TSDF)-based volumetric representation.


## ⚙️ Setup

We have ran our experiments under CUDA 9.1.85, CuDNN 7.6.5 and Ubuntu 18.04. We recommend create a virtual environment with Python 3.6 using [Anaconda](https://www.anaconda.com/download/) `conda create -n edam python=3.6` and install the dependencies as
```shell
conda install -c conda-forge opencv=4.2.0
pip3 install -r path/to/Endo-Depth-and-Motion/requirements.txt
```


## 💾 Data

The [Hamlyn](http://hamlyn.doc.ic.ac.uk/vision/) rectified images and the rectified calibration used to train and test the *Endo-Depth* models can be found [here](https://drive.google.com/drive/folders/1SYRByyAdlySvltn0CFQea1UY3AoutnKu?usp=sharing). The Hamlyn data used to test the tracking and the volumetric fusion is [here](https://drive.google.com/drive/folders/1-geZ5jJkofRd8Q3uOSOBNAHPKd0u5B2f?usp=sharing). The color and depth images are slightly cropped to avoid the small distortions of the depth *Endo-Depth* produces at the borders. The depth was computed using the stereo [Endo-Depth models](https://drive.google.com/drive/folders/17t30Jz3X-BSz-Fz7BkONqRQsOOaf5xR9?usp=sharing) and it is in [mm] and in image format uint16. The saturation depth is 300 [mm]. You can also replace it with your own data.


## 🧠 Endo-Depth

To predict the depth for a single or multiple images use
```shell
python apps/depth_estimate/__main__.py --image_path path/to/image_folder --model_path path/to/model_folder
```

You have must have already download the [Endo-Depth model](https://drive.google.com/drive/folders/17t30Jz3X-BSz-Fz7BkONqRQsOOaf5xR9?usp=sharing) you want to use. If you prefer to store the depth predictions in another folder use the argument --output_path. You can also select the type of the output with --output_type which is set by default to *grayscale* (grayscale depth images), but you can also choose *color* (colormapped depth images). By default, the saturation depth is set to *300* [mm], you can change this limit using --saturation_depth. Also, the image depth scaling is by default *52.864* because for Hamlyn dataset the weighted average baseline is 5.2864. This number is multiplied by 10 because the imposed baseline during training is 0.1. The image extension to search for in folder can be changed with --ext (now set as *jpg*), and you can disable CUDA using the argument --no_cuda.


## 👀 Tracking

You can execute our photometric tracking with
```shell
python apps/tracking_ours/__main__.py -d cuda:0 -i path/to/hamlyn_tracking_test_data -o apps/tracking_ours/results
```

being -i the input path to the folder containing the different video folders, -o the output path where the odometry in format .pkl is saved. If you want to run the script on CPU instead of on GPU just remove the argument -d *cuda:0*. The ratio frame-keyframe and number of floors of the pyramid are set to 2 by default, but they can be changed with the arguments -fr and -st, respectively. The output odometries of the Hamlyn test data using our tracking can be found [here](https://drive.google.com/drive/folders/1bcF-nrz-iWS6_mSj4fjuVBA3TvhZYRTB?usp=sharing).

To use alternatively the tracking methods of [Open3D](http://www.open3d.org/) run
```shell
python apps/tracking_open3d/__main__.py -d cuda:0 -i path/to/hamlyn_tracking_test_data -o apps/tracking_open3d/results -t park
```

The tracking method can be changed modifying the argument -t: *point-to-point* (ICP point-to-point), *point-to-plane* (ICP point-to-plane), *steinbrucker* (photometric) and *park* (hybrid photometric and geometric). Additionally, with the argument -r you can execute a global registration with RANSAC to compute a pre-translation between two point clouds before calculating the final translation with the local registration.


## ♻️ Volumetric fusion

In order to get the refined 3D map, you can fuse the registered pseudo-RGBD keyframes obtained from *Endo-Depth* and the tracking with
```shell
python apps/volumetric_fusion/__main__.py -i apps/tracking_ours/results/test1.pkl -o path/to/hamlyn_tracking_test_data/test1
```

where -i is the input odometry in format .pkl computed with the tracking. The output 3D meshes of the Hamlyn test data using the volumetric fusion are [here](https://drive.google.com/drive/folders/1sgmdtKFL1Lu8eqljKN-o_cjHRXIa7VI-?usp=sharing).


## 👩‍⚖️ License

Endo-Depth-and-Motion is released under [GPL-3.0 License](LICENSE). The code in the folder apps/depth_estimate is property of the authors of [Monodepth2](https://github.com/nianticlabs/monodepth2) and has its own [License](apps/depth_estimate/LICENSE).
