import random
import torch.utils.data as data
import os
import cv2
import numpy as np
import torch


class DatasetRESIDE(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for image-to-image mapping.
    # Both "paths_L" and "paths_H" are needed.
    # -----------------------------------------
    # e.g., train dehazing with L and H
    # -----------------------------------------
    '''

    def __init__(self, H_path='', L_path='', H_format='.png', n_channels=3, patch_size=(224, 224), is_train=False):
        super(DatasetRESIDE, self).__init__()
        print('Get L/H for image-to-image mapping. Both "paths_L" and "paths_H" are needed.')
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.H_path = H_path
        self.L_path = L_path
        self.H_format = H_format
        self.is_train = is_train

        # ------------------------------------
        # get the path of L/H
        # ------------------------------------
        self.paths_H = get_image_paths(H_path)
        self.paths_L = get_image_paths(L_path)

        assert self.paths_H, 'Error: H path is empty.'
        assert self.paths_L, 'Error: L path is empty. Plain dataset assumes both L and H are given!'

    def __getitem__(self, index):

        # ------------------------------------
        # get L image
        # ------------------------------------
        L_path = self.paths_L[index]
        img_L = imread_uint(L_path, self.n_channels)

        # ------------------------------------
        # get H image
        # ------------------------------------
        L_filename = self.paths_L[index].split('/')[-1]

        H_filename = L_filename.split("_")[0] + self.H_format

        H_path = os.path.join(self.H_path, H_filename)

        img_H = imread_uint(H_path, self.n_channels)

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.is_train:

            H, W, _ = img_H.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size[0]))
            rnd_w = random.randint(0, max(0, W - self.patch_size[1]))
            patch_L = img_L[rnd_h:rnd_h + self.patch_size[0], rnd_w:rnd_w + self.patch_size[1], :]
            patch_H = img_H[rnd_h:rnd_h + self.patch_size[0], rnd_w:rnd_w + self.patch_size[1], :]

            # # --------------------------------
            # # augmentation - flip and/or rotate
            # # --------------------------------
            # mode = random.randint(0, 7)
            # patch_L, patch_H = augment_img(patch_L, mode=mode), augment_img(patch_H, mode=mode)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_L, img_H = uint2tensor3(patch_L), uint2tensor3(patch_H)

        else:

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_L, img_H = uint2tensor3(img_L), uint2tensor3(img_H)

        return img_L, img_H

    def __len__(self):
        return len(self.paths_L)


def get_image_paths(dataroot):
    paths = None  # return None if dataroot is None
    if isinstance(dataroot, str):
        paths = sorted(_get_paths_from_images(dataroot))
    elif isinstance(dataroot, list):
        paths = []
        for i in dataroot:
            paths += sorted(_get_paths_from_images(i))
    return paths


# --------------------------------------------
# get uint8 image of size HxWxn_channles (RGB)
# --------------------------------------------
def imread_uint(path, n_channels=3):
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img


def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


# def augment_img(img, mode=0):
#     '''Kai Zhang (github: https://github.com/cszn)
#     '''
#     if mode == 0:
#         return img
#     elif mode == 1:
#         return np.flipud(np.rot90(img))
#     elif mode == 2:
#         return np.flipud(img)
#     elif mode == 3:
#         return np.rot90(img, k=3)
#     elif mode == 4:
#         return np.flipud(np.rot90(img, k=2))
#     elif mode == 5:
#         return np.rot90(img)
#     elif mode == 6:
#         return np.rot90(img, k=2)
#     elif mode == 7:
#         return np.flipud(np.rot90(img, k=3))


# convert uint to 3-dimensional torch tensor
def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


# import matplotlib.pyplot as plt

# def imshow(x, title=None, cbar=False, figsize=None):
#     plt.figure(figsize=figsize)
#     plt.imshow(np.squeeze(x), interpolation='nearest', cmap='gray')
#     if title:
#         plt.title(title)
#     if cbar:
#         plt.colorbar()
#     plt.show()
#
#
# from torchvision.utils import make_grid
# import math
#
#
# # from skimage.io import imread, imsave
# def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
#     '''
#     Converts a torch Tensor into an image Numpy array of BGR channel order
#     Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
#     Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
#     '''
#     tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # squeeze first, then clamp
#     tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
#     n_dim = tensor.dim()
#     if n_dim == 4:
#         n_img = len(tensor)
#         img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
#         img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
#     elif n_dim == 3:
#         img_np = tensor.numpy()
#         img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
#     elif n_dim == 2:
#         img_np = tensor.numpy()
#     else:
#         raise TypeError(
#             'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
#     if out_type == np.uint8:
#         img_np = (img_np * 255.0).round()
#         # Important. Unlike matlab, numpy.uint8() WILL NOT round by default.
#     return img_np.astype(out_type)


if __name__ == "__main__":

    datasets = DatasetRESIDE(H_path=r'/home/wusong/sdb/lpj/ITS/train/ITS_clear',
                             L_path=r'/home/wusong/sdb/lpj/ITS/train/ITS_haze', n_channels=3, patch_size=[128, 128],
                             is_train=True)
    from torch.utils.data import DataLoader
    print(len(datasets))
    data_loader = DataLoader(datasets, batch_size=1, shuffle=False)
    for L, H in data_loader:
        print(L)
        print(H.shape)
        break
