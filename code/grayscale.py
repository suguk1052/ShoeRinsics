import os
import os.path
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader

from code.util.option import Options
from code.util.misc import (
    make_variable,
    save_individual_images,
    get_color_mapped_images,
)
from code.util.augmentation import reverse_modification, get_image_modifications
from code.model.models import get_model
from code.dataset.masked_image_dataset import MaskedImageDataset
from code.util.evaluation import get_print


def get_average_visuals(net, image, mask, visuals=None, subtract_min_depth=True, conv=True, test_time_aug=False):
    if visuals is None:
        visuals = OrderedDict()

    if test_time_aug:
        conv = False
        images, transforms = get_image_modifications(image)
        masks, _ = get_image_modifications(mask)
    else:
        images = [image]
        masks = [mask]
        transforms = ["original"]

    depths = torch.empty((0, 1, image.shape[2], image.shape[3])).to(image.device)
    with torch.no_grad():
        for curr_image, cur_mask, transform in zip(images, masks, transforms):
            cur_outputs, _ = net(curr_image, mask=None if conv else cur_mask)
            albedo, depth, normal, light_env, light_id = cur_outputs

            cur_depth = reverse_modification(depth, transform, label='depth', original_shape=image.shape[2:])
            cur_depth[~mask] = 1
            depths = torch.cat((depths, cur_depth))

    depth_std, depth_mean = torch.std_mean(depths, dim=0, keepdim=True)
    if subtract_min_depth:
        depth_mean = depth_mean - torch.min(depth_mean)
        depth_mean[~mask.repeat(1, depth_mean.shape[1], 1, 1)] = 1
    visuals['depth pred'] = depth_mean

    return visuals

def _normalize_depth(depth, mask, lower=10, upper=90):
    """Normalize depth to [0,1] using percentiles to adjust contrast."""
    depth_np = depth.squeeze().cpu().detach().numpy()
    mask_np = mask.squeeze().cpu().detach().numpy()
    if depth_np.ndim == 3:
        depth_np = depth_np[:, :, 0]

    vals = depth_np[mask_np]
    min_ = np.percentile(vals, lower)
    max_ = np.percentile(vals, upper)
    if max_ - min_ < 1e-8:
        max_ = min_ + 1e-8
    depth_np = np.clip(depth_np, min_, max_)
    depth_np = (depth_np - min_) / (max_ - min_)
    depth_np[~mask_np] = 0

    depth_tensor = torch.tensor(depth_np).unsqueeze(0).unsqueeze(0)
    mask_tensor = torch.tensor(mask_np).unsqueeze(0).unsqueeze(0)
    return depth_tensor, mask_tensor


def enhance_depth_contrast(depth, mask, lower=10, upper=90):
    depth_norm, mask_norm = _normalize_depth(depth, mask, lower, upper)
    return depth_norm.to(depth.device), mask_norm.to(mask.device)

def prepare_datasets(opt):
    dataset_dir = os.path.join(opt.dataroot, opt.val_dataset_dir)
    dataset = MaskedImageDataset(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=opt.num_workers)
    return dataloader

def main():
    opt = Options(train=False).parse()
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    net = get_model('decomposer', weights_init=opt.weights_decomposer, output_last_ft=True, out_range='0,1').to(device)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count())))
    net.eval()

    dataloader = prepare_datasets(opt)
    image_dir = opt.val_dataset_dir + ("_test_time_aug" if opt.test_time_aug else "")
    print(f"Saving outputs to {os.path.join(opt.output, image_dir)}")

    for data in dataloader:
        image, mask, _, _, name, pad_h_before, pad_h_after, pad_w_before, pad_w_after = data
        image, mask = [make_variable(item, requires_grad=False).to(device) for item in [image, mask]]

        visuals = OrderedDict()

        visuals = get_average_visuals(
            net, image, mask, visuals=visuals, conv=False, test_time_aug=opt.test_time_aug
        )

        lower, upper = opt.depth_percentiles
        depth_norm, mask_norm = enhance_depth_contrast(
            visuals['depth pred'], mask, lower=lower, upper=upper
        )
        depth_color = get_color_mapped_images(
            depth_norm.squeeze().cpu().numpy(),
            mask_norm.squeeze().cpu().numpy(),
            mask_color=0,
            original_scale=True,
            to_tensor=True,
        ).to(device)
        depth_gray = 1 - depth_norm
        depth_gray[~mask_norm] = 0.5

        print_pred = ~get_print(depth_norm, mask_norm, None)
        print_pred = print_pred.float()
        print_pred[~mask_norm] = 0

        visuals = OrderedDict()
        visuals['print pred'] = print_pred
        visuals['depth pred'] = depth_color
        visuals['depth gray'] = depth_gray

        save_individual_images(
            visuals,
            os.path.join(opt.output, image_dir),
            name,
            pad_h_before=pad_h_before,
            pad_h_after=pad_h_after,
            pad_w_before=pad_w_before,
            pad_w_after=pad_w_after,
        )

    return

if __name__ == '__main__':
    main()
