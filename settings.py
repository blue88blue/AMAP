import os
import time
import csv
from utils import utils

class basic_setting():

    mode = "train"                             # train,  test, train_test
    k_fold = 5                            # None 不交叉验证
    start_fold = 0                        # 中断后从哪一折重新开始训练
    end_fold = 4

    # #################################### train Data settings ####################################
    dataset_file_list = "utils/AMAP_file_list.csv"
    data_root = '/home/sjh/dataset/AMAP-TECH/amap_traffic_train_0712'
    target_root = '/home/sjh/dataset/AMAP-TECH/amap_traffic_annotations_train.json'
    crop_size = (360, 640)
    num_workers = 4

    # #################################### train file settings ####################################
    run_dir = "Single_Image"                      # 数据集名称
    val_step = 5                          # 每训练几个epoch进行一次验证

    # #################################### model settings ####################################
    in_channel = 3
    n_class = 3
    network = "resnet50"  # 模型名， 或实验名称
    note = ""
    pretrained = True

    # #################################### train settings ####################################
    class_weight = [0.3, 0.3, 0.3]
    num_epochs = 100
    batch_size = 4
    lr = 0.001
    momentum = 0.9
    weight_decay = 1e-4
    # cuda_id = "0"

    # #################################### test settings ####################################
    test_run_file = '2020-0821-1116_02_resnest50__fold_None'
    label_names = ['bg', 'clear', 'slow', 'jam']
    plot = True

    def __init__(self):
        if not os.path.exists("./runs"):
            os.mkdir("./runs")
        self.run_dir = "./runs/"+self.run_dir
        if not os.path.exists(self.run_dir):
            os.mkdir(self.run_dir)

        if self.mode == "train" or self.mode == "train_test":
            time_now = time.strftime("%Y-%m%d-%H%M_%S", time.localtime(time.time()))
            # 本次训练数据文件夹 模型名+时间
            self.dir = os.path.join(self.run_dir, time_now+"_"+self.network+"_"+self.note + f'_fold_{self.k_fold}')
            os.mkdir(self.dir)

            # 训练验证指标记录文件
            self.val_result_file = os.path.join(self.dir, "val_result.csv")
            with open(self.val_result_file, "a") as f:
                w = csv.writer(f)
                w.writerow(['fold', 'epoch', 'weight_F1', 'F1', 'precision', 'recall'])

            # 日志保存文件夹
            self.log_dir = os.path.join(self.dir, "log")
            os.mkdir(self.log_dir)
            self.log_dir = [self.log_dir]

            # 模型保存文件夹
            self.checkpoint_dir = os.path.join(self.dir, "checkpoints")
            os.mkdir(self.checkpoint_dir)
            self.checkpoint_dir = [self.checkpoint_dir]

            # 交叉验证的模型保存路径
            if self.k_fold is not None:
                for i in range(self.k_fold):
                    cp_i_dir = os.path.join(self.checkpoint_dir[0], f"fold_{i+1}")
                    log_i_dir = os.path.join(self.log_dir[0], f"fold_{i+1}")
                    os.mkdir(cp_i_dir)
                    os.mkdir(log_i_dir)
                    self.checkpoint_dir.append(cp_i_dir)
                    self.log_dir.append(log_i_dir)
            self.logger(self.dir+"/train_log")

        if self.mode == "test" or self.mode == "train_test":
            if self.mode == "test":
                self.dir = os.path.join(self.run_dir, self.test_run_file)

                # 训练验证指标记录文件, 查询最优模型计算指标
                self.val_result_file = os.path.join(self.dir, "val_result.csv")

                # 模型文件夹
                self.checkpoint_dir = [os.path.join(self.dir, "checkpoints")]
                if self.k_fold is not None:
                    for i in range(self.k_fold):
                        fold_i_dir = os.path.join(self.checkpoint_dir[0], f"fold_{i+1}")
                        self.checkpoint_dir.append(fold_i_dir)
            # 测试结果文件
            self.test_result_file = os.path.join(self.dir, "test_result.csv")
            with open(self.test_result_file, "w") as f:
                w = csv.writer(f)
                title = ['file', 'mDice'] + [name+"_dice" for name in self.label_names[1:]] + \
                        ['mIoU'] + [name + "_iou" for name in self.label_names[1:]] + \
                        ['mAcc'] + \
                        ['mSens'] + [name + "_sens" for name in self.label_names[1:]] + \
                        ['mSpec'] + [name + "_spec" for name in self.label_names[1:]]
                w.writerow(title)
            # 测试结果图片保存
            if self.plot:
                self.plot_save_dir = os.path.join(self.dir, "test_images")
                if not os.path.exists(self.plot_save_dir):
                    os.mkdir(self.plot_save_dir)


    # 记录训练参数与信息
    def logger(self, file):
        with open(file, "a") as f:
            attrs = dir(self)
            for att in attrs:
                if ("__"or "test_" or "val_" or "root" or "logger" or "dir") not in att:
                    f.write(f'{str(att)}:    {str(getattr(self, att))}\n\n')
            f.close()