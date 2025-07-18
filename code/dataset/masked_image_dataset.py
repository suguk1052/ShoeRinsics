import os
import numpy as np
import cv2
from .util import read_image, image_to_channels
from code.util.misc import get_invalid_tensor

class MaskedImageDataset(object):
    """Dataset for images where background is already black.
    The mask is computed directly from the image."""

    def __init__(self, input_dir, h=128, w=128, max_width=512):
        self.input_dir = input_dir
        self.patch_height = h
        self.patch_width = w
        self.max_width = max_width

        self.image_files = sorted([
            os.path.join(self.input_dir, f)
            for f in os.listdir(self.input_dir)
            if f.lower().endswith('.jpg') or f.lower().endswith('.png')
        ])

        print("Total masked images:", len(self.image_files))


    def _compute_padding(self, shape):
        """Return padding values for an image shape."""
        h = self.patch_height
        while h < shape[0]:
            h += self.patch_height
        w = self.patch_width
        while w < shape[1]:
            w += self.patch_width

        pad_h_before = (h - shape[0]) // 2
        pad_h_after = (h - shape[0]) - pad_h_before
        pad_w_before = (w - shape[1]) // 2
        pad_w_after = (w - shape[1]) - pad_w_before
        return pad_h_before, pad_h_after, pad_w_before, pad_w_after

    def __getitem__(self, index):
        index = index % len(self.image_files)
        image = read_image(self.image_files[index])

        if image.shape[1] != self.max_width:
            scale = self.max_width / image.shape[1]
            if scale != 1:
                new_h = int(round(image.shape[0] * scale))
                image = cv2.resize(image, (self.max_width, new_h), interpolation=cv2.INTER_AREA)

        mask = np.any(image > 0, axis=2, keepdims=True)

        pad_h_before, pad_h_after, pad_w_before, pad_w_after = \
            self._compute_padding(image.shape)

        image = np.pad(image, ((pad_h_before, pad_h_after),
                               (pad_w_before, pad_w_after), (0, 0)), mode='edge')
        mask = np.pad(mask, ((pad_h_before, pad_h_after),
                             (pad_w_before, pad_w_after), (0, 0)), mode='edge')

        mask3d_inverted = ~mask.repeat(3, axis=2)
        image[mask3d_inverted] = 1
        image, mask = [image_to_channels(item) for item in [image, mask]]

        print_ = get_invalid_tensor(tensor=False)
        albedo = get_invalid_tensor(tensor=False)
        name = os.path.basename(self.image_files[index])

        return image, mask[0:1, ...].astype(bool), print_, albedo, name, \
               pad_h_before, pad_h_after, pad_w_before, pad_w_after

    def __len__(self):
        return len(self.image_files)
