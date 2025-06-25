import os
import os.path
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader

from code.util.option import Options
from code.util.misc import make_variable, save_individual_images, save_tensor_grid
from torchvision.utils import save_image
import torch.nn.functional as F
from code.util.augmentation import reverse_modification, get_image_modifications
from code.model.models import get_model
from code.dataset.masked_image_dataset import MaskedImageDataset


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

def normalize_depth_to_gray(depth, mask):
    depth = depth.squeeze().cpu().detach().numpy()
    mask = mask.squeeze().cpu().detach().numpy()
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]
    min_ = np.min(depth[mask])
    max_ = np.max(depth[mask])
    depth = (depth - min_) / (max_ - min_)
    depth[~mask] = 1
    depth = torch.tensor(depth).unsqueeze(0).unsqueeze(0)
    return depth

def crop_and_resize(tensor, pad_h_before, pad_h_after, pad_w_before, pad_w_after, width=300):
    """Crop padded regions and resize tensor to the given width preserving aspect ratio."""
    phb = int(pad_h_before)
    pha = int(pad_h_after)
    pwb = int(pad_w_before)
    pwa = int(pad_w_after)
    tensor = tensor[:, :, phb:tensor.shape[2]-pha, pwb:tensor.shape[3]-pwa]
    h, w = tensor.shape[2], tensor.shape[3]
    new_h = int(h * width / w)
    tensor = F.interpolate(tensor, size=(new_h, width), mode='bilinear', align_corners=False)
    return tensor

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
    net.eval()

    dataloader = prepare_datasets(opt)
    image_dir = opt.val_dataset_dir + ("_test_time_aug" if opt.test_time_aug else "")
    os.makedirs(os.path.join(opt.output, image_dir, "grid"), exist_ok=True)
    print(f"Saving outputs to {os.path.join(opt.output, image_dir)}")

    for data in dataloader:
        image, mask, _, _, name, pad_h_before, pad_h_after, pad_w_before, pad_w_after = data
        image, mask = [make_variable(item, requires_grad=False).to(device) for item in [image, mask]]

        visuals = OrderedDict()
        visuals[name[0]] = image
        visuals['mask'] = mask

        visuals = get_average_visuals(net, image, mask, visuals=visuals,
                                      conv=False, test_time_aug=opt.test_time_aug)

        depth_gray = normalize_depth_to_gray(visuals['depth pred'], mask)
        depth_gray = crop_and_resize(depth_gray, pad_h_before, pad_h_after, pad_w_before, pad_w_after, width=300)
        visuals['depth pred'] = depth_gray

        save_path = os.path.join(opt.output, image_dir, "grid", name[0])
        save_tensor_grid(visuals, save_path, fig_shape=[1, 3], figsize=(10, 3))

        # save only the resized grayscale depth map
        os.makedirs(os.path.join(opt.output, image_dir, "depth_gray"), exist_ok=True)
        save_image(depth_gray[0], os.path.join(opt.output, image_dir, "depth_gray", name[0]))

        del visuals[name[0]]
        visuals['real image'] = image
        save_individual_images(visuals, os.path.join(opt.output, image_dir), name,
                               pad_h_before=pad_h_before, pad_h_after=pad_h_after,
                               pad_w_before=pad_w_before, pad_w_after=pad_w_after)

    return

if __name__ == '__main__':
    main()
