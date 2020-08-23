from dataset.dataset_classifer import AMAP_Dataset_Single
from utils import utils
from settings import basic_setting
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
from utils.loss import DiceLoss, OhemCrossEntropy
import torch.nn.functional as F
from tqdm import tqdm
import csv
import random
import numpy as np
from PIL import Image
import time
import torchsummary
from torchvision.utils import save_image
# models
from model.classifer_base import classifer_base




def main(args, num_fold=0):
    # 模型选择
    model = classifer_base(backbone=args.network, pretrained_base=args.pretrained, n_class=args.n_class, in_channel=args.in_channel)

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.mode == "train" and num_fold <= 1:
        torchsummary.summary(model, (3, args.crop_size[0], args.crop_size[1]))  # #输出网络结构和参数量
    print(f'   [network: {args.network}  device: {device}]')

    if args.mode == "train":
        train(model, device, args, num_fold=num_fold)
    elif args.mode == "test":
        pass
        # if args.k_fold is not None:
        #     return test(model, device, args, num_fold=num_fold)
        # else:
        #     test(model, device, args, num_fold=num_fold)
    else:
        raise NotImplementedError





def train(model, device, args, num_fold=0):
    dataset_train = AMAP_Dataset_Single(args.data_root, args.target_root, args.crop_size, "train",
                                     k_fold=args.k_fold, num_fold=num_fold)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
    num_train_data = len(dataset_train)  # 训练数据大小

    writer = SummaryWriter(log_dir=args.log_dir[num_fold], comment=f'tb_log')
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # 定义损失函数
    # criterion = nn.CrossEntropyLoss(torch.tensor(args.class_weight, device=device))
    criterion = nn.CrossEntropyLoss()

    step = 0
    for epoch in range(args.num_epochs):
        model.train()
        lr = utils.poly_learning_rate(args, opt, epoch)  # 学习率调节

        with tqdm(total=num_train_data, desc=f'[Train] fold[{num_fold}/{args.k_fold}] Epoch[{epoch + 1}/{args.num_epochs} LR{lr:.8f}] ', unit='img') as pbar:
            for batch in dataloader_train:
                step += 1
                # 读取训练数据
                image, label = batch
                assert len(image.size()) == 4

                image = image.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.long)

                # 前向传播
                opt.zero_grad()
                pred = model(image)

                # 计算损失
                totall_loss = criterion(pred, label)

                totall_loss.backward()
                opt.step()

                if step % 5 == 0:
                    writer.add_scalar("Train/Totall_loss", totall_loss.item(), step)

                pbar.set_postfix(**{'loss': totall_loss.item()})  # 显示loss
                pbar.update(image.size()[0])


        if (epoch+1) % args.val_step == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir[num_fold], f'CP_epoch{epoch + 1}.pth'))


if __name__ == "__main__":

    # seed = 12345
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    # cudnn.benchmark = True

    args = basic_setting()
    assert args.k_fold != 1
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id

    mode = args.mode
    if args.k_fold is None:
        print("k_fold is None")
        if mode == "train_test":
            args.mode = "train"
            print("###################### Train Start ######################")
            main(args)
            args.mode = "test"
            print("###################### Test Start ######################")
            main(args)
        else:
            main(args)
    else:
        if mode == "train_test":
            print("###################### Train & Test Start ######################")

        if mode == "train" or mode == "train_test":
            args.mode = "train"
            print("###################### Train Start ######################")
            for i in range(args.start_fold, args.end_fold):
                torch.cuda.empty_cache()
                time.sleep(10)
                main(args, num_fold=i + 1)

        if mode == "test" or mode == "train_test":
            args.mode = "test"
            print("###################### Test Start ######################")
            all_dice = []
            all_iou = []
            all_acc = []
            all_sen = []
            all_spe = []
            for i in range(args.start_fold, args.end_fold):
                Dice, IoU, Acc, Sensitivity, Specificity = main(args, num_fold=i + 1)
                all_dice += Dice
                all_iou += IoU
                all_acc += Acc
                all_sen += Sensitivity
                all_spe += Specificity
            utils.save_print_score(all_dice, all_iou, all_acc, all_sen, all_spe, args.test_result_file, args.label_names)
