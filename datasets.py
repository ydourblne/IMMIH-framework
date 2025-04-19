import sys
import glob
from os.path import join
import numpy as np
from PIL import Image, ImageEnhance
import torch
import random
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as T
import config as c
from natsort import natsorted
from RGBE import *
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

use_cuda = torch.cuda.is_available()
def _normalizer_RGBE(denormalize=False):
    MEAN = [0.0, 0.0, 0.0, 0.0],
    STD = [1.0, 1.0, 1.0, 1.0],

    if denormalize:
        MEAN = [-mean / std for mean, std in zip(MEAN, STD)]
        STD = [1 / std for std in STD]

    return A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.0,)

def _normalizer_RGB(denormalize=False):
    MEAN = [0.0, 0.0, 0.0],
    STD = [1.0, 1.0, 1.0],

    if denormalize:
        MEAN = [-mean / std for mean, std in zip(MEAN, STD)]
        STD = [1 / std for std in STD]

    return A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.0,)

def rgbe_norm(rgbe):
    #return (rgbe - 127) / 128
    return rgbe / 255

def rgbe_denorm(rgbe):
    #return rgbe * 128 + 127
    return rgbe * 255

def hdr_float_norm(image):
    return image / 1e5

def hdr_float_denorm(image):
    return image * 1e5

def ldr_norm(img):
    return img / 255

def ldr_denorm(img):
    return img * 255

def numpy_to_tensor(item):
    r = torch.from_numpy(item).float()
    if r.ndim == 4:
        r = r.permute([0, 3, 1, 2])
    elif r.ndim == 3:
        r = r.permute([2, 0, 1])

    return r

def tensor_to_numpy(item):
    if item.ndim == 4:
        item = item.permute([0, 2, 3, 1])
    elif item.ndim == 3:
        item = item.permute([1, 2, 0])
    return item.cpu().numpy()

def random_crop_and_resize(image, size):

    #image = resize_image(image,size)

    h, w = image.shape[:2]

    y = np.random.randint(0, h-size)
    x = np.random.randint(0, w-size)

    image = image[y:y+size, x:x+size, :]

    return image

def resize_image(image, size, bias=5):

    image_shape = image.shape

    size_min = np.min(image_shape[:2])
    size_max = np.max(image_shape[:2])

    min_size = size + np.random.randint(1, bias)

    scale = float(min_size) / float(size_min)

    image = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale)

    return image

def LDR_transform(filename, size):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_crop = random_crop_and_resize(img, size)
    img_tensor = numpy_to_tensor(ldr_norm(img_crop))
    return img_tensor

def HDR_transform(filename, size, mode):
    img = readHdr(filename)
    if mode == "train":
        img_crop = random_crop_and_resize(img, size)
    else:
        img_crop = img
    img_tensor = numpy_to_tensor(rgbe_norm(img_crop))
    return img_tensor

class Hinet_Dataset(Dataset):
    def __init__(self, size, sec_transform=LDR_transform, cover_transform=HDR_transform, mode="train"):

        self.cover_transform = cover_transform
        self.sec_transform = sec_transform
        self.mode = mode
        self.cover_images = []
        self.sec1_images = []
        self.sec2_images = []
        self.size = size
        if self.mode == "train":
            # TRAIN SETTING
            for file in os.listdir(c.TRAIN_PATH_cover):
                self.cover_images.append(c.TRAIN_PATH_cover + '/' + file)
            for file in os.listdir(c.TRAIN_PATH_sec1):
                self.sec1_images.append(c.TRAIN_PATH_sec1 + '/' + file)
            for file in os.listdir(c.TRAIN_PATH_sec2):
                self.sec2_images.append(c.TRAIN_PATH_sec2 + '/' + file)

        if self.mode == "val":
            # VAL SETTING
            for file in os.listdir(c.VAL_PATH_cover):
                self.cover_images.append(c.VAL_PATH_cover + '/' + file)
            for file in os.listdir(c.VAL_PATH_sec1):
                self.sec1_images.append(c.VAL_PATH_sec1 + '/' + file)
            for file in os.listdir(c.VAL_PATH_sec2):
                self.sec2_images.append(c.VAL_PATH_sec2 + '/' + file)
        min_length = min(len(self.cover_images), len(self.sec1_images), len(self.sec2_images))
        self.cover_images = self.cover_images[:min_length]
        self.sec1_images = self.sec1_images[:min_length]
        self.sec2_images = self.sec2_images[:min_length]


    def __getitem__(self, index):
        cover = self.cover_transform(self.cover_images[index], self.size, self.mode)
        sec1 = self.sec_transform(self.sec1_images[index], self.size)
        sec2 = self.sec_transform(self.sec2_images[index], self.size)
        return cover, sec1, sec2

    def __len__(self):
        return len(self.cover_images)



# transform_train = A.Compose(
#         [
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.RandomCrop(height=c.cropsize_train, width=c.cropsize_train)
#         ],
#     )
# transform_val = A.Compose(
#         [
#             A.CenterCrop(height=c.cropsize_val, width=c.cropsize_val)
#         ],
#     )


# Training data loader
trainloader = DataLoader(
    Hinet_Dataset(size=c.cropsize_train,mode="train"),
    batch_size=c.batchsize_train,
    shuffle=True,  # 打乱数据
    pin_memory=False,
    num_workers=1,
    drop_last=True
)
# Test data loader
testloader = DataLoader(
    Hinet_Dataset(size=c.cropsize_val, mode="val"),
    batch_size=c.batchsize_val,
    shuffle=False,
    pin_memory=False,
    num_workers=1,
    drop_last=True
)



