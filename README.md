# Endo-Depth-and-Motion

This repository contains the tracking and volumetric fusion code of the methods used in

> **Localization and Reconstruction in Endoscopic Videos using Depth Networks and Photometric Constraints**
>
> [David Recasens](https://davidrecasens.github.io/), [José Lamarca](https://webdiis.unizar.es/~jlamarca/), [José M. Fácil](https://webdiis.unizar.es/~jmfacil/), [José María M. Montiel](https://janovas.unizar.es/sideral/CV/jose-maria-martinez-montiel) and [Javier Civera](https://janovas.unizar.es/sideral/CV/javier-civera-sancho)

<p align="center">
  <img src="assets/teaser.gif" alt="example input output gif" width="600" />
</p>


## ⚙️ Setup

We have ran our experiments under CUDA 9.1.85, CuDNN 7.6.5 and Ubuntu 18.04. We recommend create a virtual environment with Python 3.6 using [Anaconda](https://www.anaconda.com/download/) `conda create -n edam python=3.6` and install the dependencies as:
```shell
conda install -c conda-forge opencv=4.2.0
pip3 install -r path/to/Endo-Depth-and-Motion/requirements.txt
```


## 💾 Test data

The [Hamlyn](http://hamlyn.doc.ic.ac.uk/vision/) data used to test the tracking and the volumetric fusion can be found [here](https://drive.google.com/drive/folders/1-geZ5jJkofRd8Q3uOSOBNAHPKd0u5B2f?usp=sharing). The color and depth images are little cropped to avoid the small distortions of the depth Endo-Depth produces at the borders. The depth was computed using the stereo [Endo-Depth models](https://drive.google.com/drive/folders/17t30Jz3X-BSz-Fz7BkONqRQsOOaf5xR9?usp=sharing). You can also replace it with your own data.


## 👀 Tracking

You can execute our photometric tracking with
```shell
python apps/tracking_ours/__main__.py -d cuda:0 -i path/to/hamlyn_tracking_test_data -o apps/tracking_ours/results
```

being -i the input path to the folder containing the different video folders, -o the output path where the odometry in format .pkl is saved. If you want to run the script on CPU instead of on GPU just remove the argument -d cuda:0. The ratio frame-keyframe and number of floors of the pyramid are set to 2 by default, but they can be changed with the arguments -fr and -st, respectively. The output odometries of the Hamlyn test data using our tracking can be found [here](https://drive.google.com/drive/folders/1bcF-nrz-iWS6_mSj4fjuVBA3TvhZYRTB?usp=sharing).

To use the tracking methods of [Open3D](http://www.open3d.org/) run
```shell
python apps/tracking_open3d/__main__.py -d cuda:0 -i path/to/hamlyn_tracking_test_data -o apps/tracking_open3d/results -t park
```

The tracking method can be changed modifying the argument -t: point-to-point (ICP point-to-point), point-to-plane (ICP point-to-plane), steinbrucker (photometric) and park (hybrid photometric and geometric). Additionally, with the argument -r you can execute a global registration with RANSAC to compute a pre-translation between two point clouds before calculating the final translation with the local registration.


## ♻️ Volumetric fusion

In order to get the refined 3D map, you can fuse the registered pseudo-RGBD keyframes obtained from Endo-Depth and the tracking with
```shell
python apps/volumetric_fusion/__main__.py -i apps/tracking_ours/results/test1.pkl -o path/to/hamlyn_tracking_test_data/test1
```

where -i is the input odometry in format .pkl computed with the tracking. The output 3D meshes of the Hamlyn test data using the volumetric fusion can be found [here](https://drive.google.com/drive/folders/1sgmdtKFL1Lu8eqljKN-o_cjHRXIa7VI-?usp=sharing).
