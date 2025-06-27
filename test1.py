import os
import argparse
import cv2
import numpy as np
import subprocess


def create_dataset(image_path, dataroot, dataset_name):
    dataset_dir = os.path.join(dataroot, dataset_name)
    image_dir = os.path.join(dataset_dir, 'image')
    mask_dir = os.path.join(dataset_dir, 'mask')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f'Cannot read image: {image_path}')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    img_name = os.path.basename(image_path)
    cv2.imwrite(os.path.join(image_dir, img_name), img)
    cv2.imwrite(os.path.join(mask_dir, img_name), mask)

    return dataset_name


def run_test(weights_decomposer, dataroot, dataset_name, output, extra_opts=None):
    cmd = [
        'python', 'code/test.py',
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
    parser.add_argument('--image_path', required=True, help='Masked shoe image with black background')
    parser.add_argument('--weights_decomposer', required=True, help='Path to decomposer weights')
    parser.add_argument('--dataroot', default='single_data', help='Temporary dataset root folder')
    parser.add_argument('--dataset_name', default='single_test', help='Directory name inside dataroot')
    parser.add_argument('--output', default='../results', help='Directory to store results')
    parser.add_argument('--extra_opts', default='', help='Additional options for test.py')
    args = parser.parse_args()

    dataset_name = create_dataset(args.image_path, args.dataroot, args.dataset_name)
    run_test(args.weights_decomposer, args.dataroot, dataset_name, args.output, args.extra_opts)


if __name__ == '__main__':
    main()
