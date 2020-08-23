import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw, PIL.ImageFilter
import os


def random_Rotate(img, label=None):
    rand = int(float(torch.rand(1)-0.5)*60)
    img = img.rotate(rand)
    if label is not None:
        label = label.rotate(rand)
    return img, label


def random_Contrast(img):
    v = float(torch.rand(1)) * 2
    if 0.5 <= v <= 1.5:
        return PIL.ImageEnhance.Contrast(img).enhance(v)
    else:
        return img

# 颜色鲜艳或灰暗
def random_Color(img):
    v = float(torch.rand(1)) * 2
    if 0.4 <= v <= 1.5:
        return PIL.ImageEnhance.Color(img).enhance(v)
    else:
        return img

# 亮度变换
def random_Brightness(img):  # [0.1,1.9]
    v = float(torch.rand(1)) * 2
    if 0.4 <= v <= 1.5:
        return PIL.ImageEnhance.Brightness(img).enhance(v)
    else:
        return img


def random_GaussianBlur(img):
    p = float(torch.rand(1))
    if p > 0.6:
        v = float(torch.rand(1))+1.2
        return img.filter(PIL.ImageFilter.GaussianBlur(radius=v))
    else:
        return img


def random_transfrom(image, label=None):
    image = random_Color(image)
    # image, label = random_Rotate(image, label)
    image = random_Contrast(image)
    image = random_Brightness(image)
    image = random_GaussianBlur(image)
    return image, label

# 读取图片与mask，返回4D张量
def fetch(image_path, label_path=None):
    image = Image.open(image_path)

    if label_path is not None:
        label = Image.open(label_path)
    else:
        label = None

    return image, label

# image转为tensor
def convert_to_tensor(image, label=None):
    image = torch.FloatTensor(np.array(image)) / 255
    # image = image - torch.Tensor(np.array([0.5, 0.5, 0.5]))
    if len(image.size()) == 2:
        image = image.unsqueeze(0)
    image = image.permute(2, 0, 1)

    if label is not None:
        label = torch.FloatTensor(np.array(label))
        if len(label.size()) == 2:
            label = label.unsqueeze(0)
    return image, label

def scale(crop_size, image, label=None):

    image = F.interpolate(image.unsqueeze(0), size=crop_size, mode='bilinear', align_corners=True).squeeze(0)
    if label is not None:
        label = F.interpolate(label.unsqueeze(0), size=crop_size, mode='nearest').squeeze(0)
    return image, label


# 原图小于裁剪图，填充0
def pad(crop_size, image, label=None, pad_value=0.0):
    h, w = image.size()[-2:]
    pad_h = max(crop_size[0] - h, 0)
    pad_w = max(crop_size[1] - w, 0)
    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=pad_value)
        if label is not None:
            label = F.pad(label, (0, pad_w, 0, pad_h), mode='constant', value=pad_value)
    return image, label

# 随机垂直翻转
def random_Top_Bottom_filp(image, label=None, p=0.5):
    a = float(torch.rand(1))
    if a > p:
        image = torch.flip(image, [1])
        if label is not None:
            if len(label.size()) == 2:
                label = label.unsqueeze(0)
            label = torch.flip(label, [1])
    return image, label

# 随机垂直翻转
def random_Left_Right_filp(image, label=None, p=0.5):
    a = float(torch.rand(1))
    if a > p:
        image = torch.flip(image, [2])
        if label is not None:
            if len(label.size()) == 2:
                label = label.unsqueeze(0)
            label = torch.flip(label, [2])
    return image, label
