import torch
import os
from torch.utils.data import Dataset
import csv
from .transform import*
import numpy as np
import json
from utils.utils import get_dataset_filelist

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



class AMAP_Dataset_Single(Dataset):
    def __init__(self, data_root, annotations_root, crop_size, data_mode, k_fold=None, num_fold=None):
        super().__init__()
        self.crop_size = crop_size  # h, w
        self.data_root = data_root
        self.data_mode = data_mode

        all_train_files = get_dataset_filelist(annotations_root)  # 所有图片的相对路径

        if k_fold == None:
            self.image_files = all_train_files
            print(f"{data_mode} dataset: {len(self.image_files)}")
        # 交叉验证：传入包含所有数据集文件名的csv， 根据本次折数num_fold获取文件名列表
        else:
            image_files = all_train_files
            fold_size = len(image_files) // k_fold  # 等分
            fold = num_fold - 1
            if data_mode == "train":
                self.image_files = image_files[0: fold*fold_size] + image_files[fold*fold_size+fold_size:]
            elif data_mode == "val" or data_mode == "test":
                self.image_files = image_files[fold*fold_size: fold*fold_size+fold_size]
            else:
                raise NotImplementedError
            print(f"{data_mode} dataset fold{num_fold}/{k_fold}: {len(self.image_files)} images")

        # print(self.image_files)

        with open(annotations_root, "r") as f:
            data = json.load(f)
            self.label_file = data["annotations"]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file = self.image_files[idx]
        image_path = os.path.join(self.data_root, file)
        image, _ = fetch(image_path)

        if self.data_mode == "train":  # 数据增强
            image = random_Color(image)
            image = random_Contrast(image)
            image = random_Brightness(image)

        image, _ = convert_to_tensor(image)
        # -------标签处理-------
        index = int(os.path.split(file)[0])-1
        label = int(self.label_file[index].get("status"))
        label = torch.tensor(label)
        # -------标签处理-------

        if self.data_mode == "train":  # 数据增强
            image, _ = random_Left_Right_filp(image)

        image, _ = scale(self.crop_size, image)

        if self.data_mode == "val":  # 若测试数据， 同时输出文件名
            return image, label, index
        return image, label




class AMAP_PredictDataset(Dataset):
    def __init__(self, data_root, annotations_root, crop_size):
        super(AMAP_PredictDataset, self).__init__()
        self.data_root = data_root
        self.crop_size = crop_size

        self.files = get_dataset_filelist(annotations_root)
        print(f"pred dataset: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        id = int(os.path.split(file)[0])
        image_path = os.path.join(self.data_root, file)

        image, _ = fetch(image_path)
        image, _ = convert_to_tensor(image)
        image, _ = scale(self.crop_size, image)

        return image, id



