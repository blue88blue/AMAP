from dataset.dataset_single import AMAP_PredictDataset
from torch.utils.data import DataLoader
import torch
import os
from PIL import Image
from torch.nn import functional as F
from settings import basic_setting
import numpy as np
from tqdm import tqdm
import json
from dataset.dataset_single import AMAP_PredictDataset
from dataset.transform import *
import csv
#models
from model.classifer_base import classifer_base


# #################################### predict settings 预测提交结果 ####################################
pred_data_root = '/home/sjh/dataset/AMAP-TECH/amap_traffic_test_0712'
pred_annotations_root = '/home/sjh/dataset/AMAP-TECH/amap_traffic_annotations_test.json'
pred_dir = 'amap_traffic_annotations_test_key.json'
model_CPepoch = 40
# #################################### predict settings 预测提交结果 ####################################


def pred_single(model, device, args):
    with open(pred_annotations_root, 'r') as f:
        data = json.load(f)
    dataset_pred = AMAP_PredictDataset(pred_data_root, pred_annotations_root, args.crop_size)
    dataloader_pred = DataLoader(dataset_pred, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)

    all_pred_status = np.zeros((len(data["annotations"]), args.n_class), dtype=np.int)
    model.eval()
    with tqdm(total=len(dataset_pred), desc=f'predict', unit='id') as pbar:
        for batch in dataloader_pred:
            # 读取数据
            image, id = batch
            image = image.to(device, dtype=torch.float32)

            with torch.no_grad():
                pred = model(image)
                pred = torch.softmax(pred, dim=1)
                pred_status = pred.max(dim=1)[1]

            for i in range(image.size()[0]):
                index = id[i] - 1
                id_status = pred_status[i].item()
                all_pred_status[index, id_status] += 1
            pbar.update(image.size()[0])

    for i in range(len(data["annotations"])):
        status = np.argmax(np.bincount(all_pred_status[i, :]))
        data["annotations"][i]["status"] = int(status)

    # 保存预测结果
    res2 = json.dumps(data, indent=4)
    with open(pred_dir, 'w', encoding='utf-8') as f:  # 打开文件
        f.write(res2)  # 在文件里写入转成的json串





if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = basic_setting()

    # 模型选择
    model = classifer_base(backbone=args.network, pretrained_base=args.pretrained, n_class=args.n_class, in_channel=args.in_channel)

    pred_dir = os.path.join(args.dir, pred_dir)  # 预测文件

    model.to(device)
    model_dir = os.path.join(args.checkpoint_dir[0], f'CP_epoch{model_CPepoch}.pth')  # 最后一个epoch模型
    model.load_state_dict(torch.load(model_dir, map_location=device))
    print("model loaded!")

    pred_single(model, device, args)



