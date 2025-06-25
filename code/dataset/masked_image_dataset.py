import os
import numpy as np
from .util import read_image, image_to_channels
from code.util.misc import get_invalid_tensor

class MaskedImageDataset(object):
    """Dataset for images where background is already black.
    The mask is computed directly from the image."""

    def __init__(self, input_dir, h=128, w=128):
        self.input_dir = input_dir
        self.patch_height = h
        self.patch_width = w

        self.image_files = sorted([
            os.path.join(self.input_dir, f)
            for f in os.listdir(self.input_dir)
            if f.lower().endswith('.jpg') or f.lower().endswith('.png')
        ])

        print("Total masked images:", len(self.image_files))

        sample_image = read_image(self.image_files[0])
        shape = sample_image.shape
        h = self.patch_height
        while h < shape[0]:
            h += self.patch_height
        w = self.patch_width
        while w < shape[1]:
            w += self.patch_width

        self.pad_h_before = (h - shape[0]) // 2
        self.pad_h_after = (h - shape[0]) - self.pad_h_before
        self.pad_w_before = (w - shape[1]) // 2
        self.pad_w_after = (w - shape[1]) - self.pad_w_before

    def __getitem__(self, index):
        index = index % len(self.image_files)
        image = read_image(self.image_files[index])

        mask = np.any(image > 0, axis=2, keepdims=True)
        image = np.pad(image, ((self.pad_h_before, self.pad_h_after),
                               (self.pad_w_before, self.pad_w_after), (0, 0)), mode='edge')
        mask = np.pad(mask, ((self.pad_h_before, self.pad_h_after),
                             (self.pad_w_before, self.pad_w_after), (0, 0)), mode='edge')

        mask3d_inverted = ~mask.repeat(3, axis=2)
        image[mask3d_inverted] = 1
        image, mask = [image_to_channels(item) for item in [image, mask]]

        print_ = get_invalid_tensor(tensor=False)
        albedo = get_invalid_tensor(tensor=False)
        name = os.path.basename(self.image_files[index])

        return image, mask[0:1, ...].astype(np.bool), print_, albedo, name, \
               self.pad_h_before, self.pad_h_after, self.pad_w_before, self.pad_w_after

    def __len__(self):
        return len(self.image_files)
