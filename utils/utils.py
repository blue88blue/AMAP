
import torch
import math
from .loss import BinaryDiceLoss
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import csv
import os
import random
import h5py
import json


def fast_hist(label_true, label_pred, n_class):
    '''
    :param label_true: 0 ~ n_class (batch, h, w)
    :param label_pred: 0 ~ n_class (batch, h, w)
    :param n_class: 类别数
    :return: 对角线上是每一类分类正确的个数，其他都是分错的个数
    '''

    assert n_class > 1

    mask = (label_true >= 0) & (label_true < n_class)
    hist = torch.bincount(
        n_class * label_true[mask].int() + label_pred[mask].int(),
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)

    return hist

# 计算指标
def cal_scores(hist, smooth=1):
    TP = np.diag(hist)
    FP = hist.sum(axis=0) - TP
    FN = hist.sum(axis=1) - TP
    TN = hist.sum() - TP - FP - FN
    union = TP + FP + FN

    dice = (2*TP+smooth) / (union+TP+smooth)

    iou = (TP+smooth) / (union+smooth)

    Precision = np.diag(hist).sum() / hist.sum()   # 分类正确的准确率  acc

    Sensitivity = (TP+smooth) / (TP+FN+smooth)  # recall

    Specificity = (TN+smooth) / (FP+TN+smooth)

    return dice[1:]*100, iou[1:]*100, Precision*100, Sensitivity[1:]*100, Specificity[1:]*100  # 不包含背景


def cal_classifer_scores(hist, weight=[0.2, 0.2, 0.6]):
    TP = np.diag(hist)
    FP = hist.sum(axis=0) - TP
    FN = hist.sum(axis=1) - TP
    TN = hist.sum() - TP - FP - FN

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = (2 * precision * recall)/(precision + recall)

    weight_F1 = np.array(weight) * F1

    return precision.mean(), recall.mean(), F1.mean(), weight_F1.sum()

# 保存打印指标
def save_print_score(all_dice, all_iou, all_acc, all_sen, all_spe, file, label_names):
    all_dice = np.array(all_dice)
    all_iou = np.array(all_iou)
    all_acc = np.array(all_acc)
    all_sen = np.array(all_sen)
    all_spe = np.array(all_spe)
    test_mean = ["mean"]+[all_dice.mean()] + list(all_dice.mean(axis=0)) + \
                [all_iou.mean()] + list(all_iou.mean(axis=0)) + \
                [all_acc.mean()] + \
                [all_sen.mean()] + list(all_sen.mean(axis=0)) + \
                [all_spe.mean()] + list(all_spe.mean(axis=0))
    test_std = ["std"]+[all_dice.std()] + list(all_dice.std(axis=0)) + \
               [all_iou.std()] + list(all_iou.std(axis=0)) + \
               [all_acc.std()] + \
               [all_sen.std()] + list(all_sen.std(axis=0)) + \
               [all_spe.std()] + list(all_spe.std(axis=0))
    title = [' ', 'mDice'] + [name + "_dice" for name in label_names[1:]] + \
            ['mIoU'] + [name + "_iou" for name in label_names[1:]] + \
            ['mAcc'] + \
            ['mSens'] + [name + "_sen" for name in label_names[1:]] + \
            ['mSpec'] + [name + "_spe" for name in label_names[1:]]
    with open(file, "a") as f:
        w = csv.writer(f)
        w.writerow(["Test Result"])
        w.writerow(title)
        w.writerow(test_mean)
        w.writerow(test_std)

    print("\n##############Test Result##############")
    print(f'mDice: {all_dice.mean()}')
    print(f'mIoU:  {all_iou.mean()}')
    print(f'mAcc:  {all_acc.mean()}')
    print(f'mSens: {all_sen.mean()}')
    print(f'mSpec: {all_spe.mean()}')



# 从验证指标中选择最优的epoch
def best_model_in_fold(val_result, num_fold):
    best_epoch = 0
    best_dice = 0
    for row in val_result:
        if str(num_fold) in row:
            if best_dice < float(row[2]):
                best_dice = float(row[2])
                best_epoch = int(row[1])
    return best_epoch



def make_class_label(mask):
    b, h, w = mask.size()
    mask = mask.view(b, -1)
    class_label = torch.max(mask, dim=-1)[0]
    return class_label








# 读取数据集目录内文件名
def get_dataset_filelist(data_root):
    with open(data_root, 'r') as f:
        data = json.load(f)
        data = data["annotations"]

    file_list = []
    for sample in data:
        id = sample.get("id")
        frames = sample.get("frames")
        for frame in frames:
            file_list.append(os.path.join(id, frame.get("frame_name")))

    return file_list


# 将相邻的图片序列作为图片对
def make_pairs(data_root):
    with open(data_root, 'r') as f:
        data = json.load(f)
        data = data["annotations"]

    pairs = []
    for sample in data:
        id = sample["id"]
        status = int(sample["status"])
        for i in range(len(sample["frames"]) - 1):
            frame_name1 = sample["frames"][i]["frame_name"]
            time1 = int(sample["frames"][i]["gps_time"])
            frame_name2 = sample["frames"][i + 1]["frame_name"]
            time2 = int(sample["frames"][i + 1]["gps_time"])

            if math.fabs(time2 - time1) < 300:
                pair = [id, status, frame_name1, frame_name2, time2 - time1]
                pairs.append(pair)
    return pairs



class DiceCoeff(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, predict, target):
        total_loss = torch.zeros(predict.size()[1])  # 存放各个类的dice

        target = make_one_hot(target, predict.size()[1])  # 转换成onehot

        dice = BinaryDiceLoss(**self.kwargs)
        predict = F.softmax(predict, dim=1)
        # 二维图像展平
        predict = predict.contiguous().view(predict.size()[0], predict.size()[1], -1)

        for i in range(0, target.shape[1]):
            # 计算每个类别的dice
            total_loss[i] = dice(predict[:, i, :], target[:, i, :])

        return total_loss



def poly_learning_rate(args, optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs(step = 30)
    """
    lr = args.lr * (1 - epoch / args.num_epochs) ** 0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr




# 把0,1,2...这样的类别标签转化为one_hot
def make_one_hot(targets, num_classes):
    # 二维图像展平
    targets = targets.contiguous().view(targets.size()[0], 1,-1)
    for j in range(num_classes):
        target_j = torch.where(targets == j, torch.ones_like(targets), torch.zeros_like(targets))
        if j == 0:
            result = target_j
        else:
            result = torch.cat((result, target_j), dim=1)

    return result





def save_h5(train_data, train_label, val_data, filename):
    file = h5py.File(filename, 'w')
    # 写入
    file.create_dataset('train_data', data=train_data)
    file.create_dataset('train_label', data=train_label)
    file.create_dataset('val_data', data=val_data)
    file.close()


def load_h5(path):
    file = h5py.File(path, 'r')
    train_data = torch.tensor(np.array(file['train_data'][:]))
    train_label = torch.tensor(np.array(file['train_label'][:]))
    val_data = torch.tensor(np.array(file['val_data'][:]))
    file.close()
    return train_data, train_label, val_data




