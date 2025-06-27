"""Helper to run ``code.test`` on masked shoe images.

This script converts a single image or a directory of images with black
backgrounds into the dataset layout expected by ``code.test`` and then runs
the evaluation.

Example::

    python test1.py --weights_decomposer models/decomposer_best_state.t7 \
        --input path/to/masked_images
"""

import os
import argparse
import cv2
import numpy as np
import subprocess


def create_dataset(input_path, dataroot, dataset_name):
    """Create a temporary dataset from a single image or a directory."""
    dataset_dir = os.path.join(dataroot, dataset_name)
    image_dir = os.path.join(dataset_dir, 'image')
    mask_dir = os.path.join(dataset_dir, 'mask')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    if os.path.isdir(input_path):
        image_paths = [
            os.path.join(input_path, f)
            for f in sorted(os.listdir(input_path))
            if f.lower().endswith(('.jpg', '.png'))
        ]
    else:
        image_paths = [input_path]

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f'Cannot read image: {img_path}')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        img_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(image_dir, img_name), img)
        cv2.imwrite(os.path.join(mask_dir, img_name), mask)

    return dataset_name


def run_test(weights_decomposer, dataroot, dataset_name, output, extra_opts=None):
    cmd = [
        'python', '-m', 'code.test',
        f'--weights_decomposer={weights_decomposer}',
        f'--dataroot={dataroot}',
        f'--val_dataset_dir={dataset_name}',
        f'--output={output}'
    ]
    if extra_opts:
        cmd += extra_opts.split()
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description='Test a single masked shoe image.')
    parser.add_argument('--input', required=True,
                        help='Path to a masked shoe image or a directory of images')
    parser.add_argument('--weights_decomposer', required=True, help='Path to decomposer weights')
    parser.add_argument('--dataroot', default='single_data', help='Temporary dataset root folder')
    parser.add_argument('--dataset_name', default='single_test', help='Directory name inside dataroot')
    parser.add_argument('--output', default='../results', help='Directory to store results')
    parser.add_argument('--extra_opts', default='', help='Additional options for test.py')
    args = parser.parse_args()

    dataset_name = create_dataset(args.input, args.dataroot, args.dataset_name)
    run_test(args.weights_decomposer, args.dataroot, dataset_name, args.output, args.extra_opts)


if __name__ == '__main__':
    main()
