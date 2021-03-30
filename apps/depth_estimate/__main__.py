# Modified test_function of Monodepth2 (https://github.com/nianticlabs/monodepth2). Made by David Recasens.
#
# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import time
import torch
from torchvision import transforms

from resnet_encoder import ResnetEncoder
from depth_decoder import DepthDecoder


def parse_args():
    parser = argparse.ArgumentParser(
        description='Function to estimate depth maps of single or multiple images using an Endo-Depth model.')

    parser.add_argument(
        '--image_path',
        type=str,
        help='path to a test image or folder of images',
        required=True
    )
    parser.add_argument(
        '--ext',
        type=str,
        help='image extension to search for in folder',
        default="jpg"
    )
    parser.add_argument(
        "--no_cuda",
        help='if set, disables CUDA',
        action='store_true'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help='path to the model to load'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        help='path to the output directory where the results will be stored'
    )
    parser.add_argument(
        '--scale',
        type=int,
        default=52.864,
        help='image depth scaling. For Hamlyn dataset the weighted average baseline is 5.2864. It is multiplied by 10 '
             'because the imposed baseline during training is 0.1',
    )
    parser.add_argument(
        '--output_type',
        type=str,
        default='grayscale',
        choices=['grayscale', 'color'],
        help='type of the output: grayscale depth images (grayscale) or colormapped depth images (color)',
    )
    parser.add_argument(
        '--saturation_depth',
        type=int,
        default=300,
        help='saturation depth of the estimated depth images. For Hamlyn dataset it is 300 mm by default',
    )

    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_path is not None, \
        "You must specify the --model_path parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("-> Loading model from ", args.model_path)
    encoder_path = os.path.join(args.model_path, "encoder.pth")
    depth_decoder_path = os.path.join(args.model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # Extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        if args.output_path:
            output_path = args.output_path
        else:
            output_path = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = sorted(glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext))))
        if args.output_path:
            output_path = args.output_path
        else:
            output_path = args.image_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # Predicting on each image in turn
    with torch.no_grad():
        for idx, image_path in enumerate(paths):
            begin = time.time()
            if image_path.endswith("_depth.jpg") or image_path.endswith("_depth.png"):
                # don't try to predict depth for a depth image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # Prediction
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            _, scaled_depth = disp_to_depth(disp_resized_np, 0.1, 100)  # Scaled depth
            depth = scaled_depth * args.scale  # Metric scale (mm)
            depth[depth > args.saturation_depth] = args.saturation_depth
            end = time.time()

            if args.output_type == 'color':
                # Saving colormapped depth image
                vmin = depth.min()
                vmax = np.percentile(depth, 99)
                normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                mapper = cm.ScalarMappable(norm=normalizer, cmap='magma_r')
                colormapped_im = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)
                im = pil.fromarray(colormapped_im)
                output_name = os.path.splitext(os.path.basename(image_path))[0]
                output_file = os.path.join(output_path, "{}_depth.jpg".format(output_name))
                im.save(output_file)
            else:
                # Saving grayscale depth image
                im_depth = depth.astype(np.uint16)
                im = pil.fromarray(im_depth)
                output_name = os.path.splitext(os.path.basename(image_path))[0]
                output_file = os.path.join(output_path, "{}_depth.png".format(output_name))
                im.save(output_file)

            time_one_image = round((end-begin) * 1000)
            print("   Processed {:d} of {:d} images - saved prediction to {} - Computing time {}ms".format(
                idx + 1, len(paths), output_file, time_one_image))

    print('\n-> Done!')


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
