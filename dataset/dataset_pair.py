import torch
import os
from torch.utils.data import Dataset
import csv
from .transform import*
import numpy as np
import json
from utils.utils import make_pairs

# 随机擦除
def random_erase(image):
    h, w = image.size()[-2:]
    point1 = torch.rand(2)*0.8  # 左上角点
    point2 = point1 + torch.rand(2)*0.8+0.05  # 右下角点
    point2 = torch.where(point2>1, torch.ones_like(point2), point2)

    s_h = int(point1[0] * h)
    s_w = int(point1[1] * w)
    e_h = int(point2[0] * h)
    e_w = int(point2[1] * w)

    image[:, s_h: e_h, s_w: e_w] = 0
    return image

# image转为tensor
def convert_to_tensor_pair(image, label):
    image = torch.FloatTensor(np.array(image)) / 255
    if len(image.size()) == 2:
        image = image.unsqueeze(0)
    image = image.permute(2, 0, 1)

    label = torch.FloatTensor(np.array(label)) / 255
    if len(label.size()) == 2:
        label = label.unsqueeze(0)
    label = label.permute(2, 0, 1)
    return image, label


class AMAP_Dataset_Pair(Dataset):
    def __init__(self, data_root, annotations_root, crop_size, data_mode, k_fold=None, num_fold=None):
        super().__init__()
        self.crop_size = crop_size  # h, w
        self.data_root = data_root
        self.data_mode = data_mode

        all_train_files = make_pairs(annotations_root)  # 所有图片的相对路径

        if k_fold == None:
            self.image_pairs = all_train_files
            print(f"{data_mode} dataset: {len(self.image_pairs)}")
        # 交叉验证：传入包含所有数据集文件名的csv， 根据本次折数num_fold获取文件名列表
        else:
            image_files = all_train_files
            fold_size = len(image_files) // k_fold  # 等分
            fold = num_fold - 1
            if data_mode == "train":
                self.image_pairs = image_files[0: fold*fold_size] + image_files[fold*fold_size+fold_size:]
            elif data_mode == "val" or data_mode == "test":
                self.image_pairs = image_files[fold*fold_size: fold*fold_size+fold_size]
            else:
                raise NotImplementedError
            print(f"{data_mode} dataset fold{num_fold}/{k_fold}: {len(self.image_pairs)} images")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        pair = self.image_pairs[idx]
        id = pair[0]
        label = torch.tensor(pair[1])
        image_file1 = pair[2]
        image_file2 = pair[3]
        time_diff = torch.tensor([float(pair[4])/20])

        image_path1 = os.path.join(self.data_root, os.path.join(id, image_file1))
        image_path2 = os.path.join(self.data_root, os.path.join(id, image_file2))
        image1, image2 = fetch(image_path1, image_path2)

        if self.data_mode == "train":  # 数据增强
            image1, _ = random_transfrom(image1)
            image2, _ = random_transfrom(image2)

        image1, image2 = convert_to_tensor_pair(image1, image2)

        if self.data_mode == "train":  # 数据增强
            image1, image2 = random_Left_Right_filp(image1, image2)

        image1 = F.interpolate(image1.unsqueeze(0), size=self.crop_size, mode='bilinear', align_corners=True).squeeze(0)
        image2 = F.interpolate(image2.unsqueeze(0), size=self.crop_size, mode='bilinear', align_corners=True).squeeze(0)

        return image1, image2, label, time_diff




class AMAP_PredictDataset_Pair(Dataset):
    def __init__(self, data_root, annotations_root, crop_size):
        super(AMAP_PredictDataset_Pair, self).__init__()
        self.data_root = data_root
        self.crop_size = crop_size

        self.pairs = make_pairs(annotations_root)  # 所有图片的相对路径
        print(f"pred dataset: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pair = self.image_pairs[idx]
        id = pair[0]
        image_file1 = pair[2]
        image_file2 = pair[3]
        time_diff = torch.tensor([float(pair[4])/20])

        image_path1 = os.path.join(self.data_root, os.path.join(id, image_file1))
        image_path2 = os.path.join(self.data_root, os.path.join(id, image_file2))
        image1, image2 = fetch(image_path1, image_path2)
        image1, image2 = convert_to_tensor_pair(image1, image2)
        image1 = F.interpolate(image1.unsqueeze(0), size=self.crop_size, mode='bilinear', align_corners=True).squeeze(0)
        image2 = F.interpolate(image2.unsqueeze(0), size=self.crop_size, mode='bilinear', align_corners=True).squeeze(0)

        return image1, image2, id, time_diff
