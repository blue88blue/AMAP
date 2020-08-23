from dataset.dataset_classifer import AMAP_PredictDataset
from torch.utils.data import DataLoader
import torch
import os
from PIL import Image
from torch.nn import functional as F
from settings import basic_setting
import numpy as np
from tqdm import tqdm
import json
from dataset.transform import *
import csv
#models
from model.classifer_base import classifer_base


# #################################### predict settings 预测提交结果 ####################################
pred_data_root = '/home/sjh/dataset/AMAP-TECH/amap_traffic_test_0712'
pred_annotations_root = '/home/sjh/dataset/AMAP-TECH/amap_traffic_annotations_test.json'
pred_dir = 'amap_traffic_annotations_test_key.json'
model_CPepoch = 20
# #################################### predict settings 预测提交结果 ####################################


def pred_single(model, device, args):
    with open(pred_annotations_root, 'r') as f:
        data = json.load(f)
    num_data = len(data["annotations"])

    model.eval()
    with tqdm(total=num_data, desc=f'predict', unit='id') as pbar:
        for i in range(num_data):
            id = data["annotations"][i]["id"]
            id_path = os.path.join(pred_data_root, id)
            frames = data["annotations"][i]["frames"]

            id_pred_status = []
            key_frame_status = -1
            for j in range(len(frames)):
                # 按id 读取图片
                image_file = frames[j]["frame_name"]
                if image_file == data["annotations"][i]["key_frame"]:
                    key_frame = True
                else:
                    key_frame = False
                image_path = os.path.join(id_path, image_file)
                image, _ = fetch(image_path)
                image, _ = convert_to_tensor(image)
                image, _ = scale(args.crop_size, image)
                image = image.to(device, dtype=torch.float32)
                image = image.unsqueeze(0)  # (1, 3, h, w)

                with torch.no_grad():
                    pred = model(image)
                    pred = torch.softmax(pred, dim=1)
                    pred_status = pred.max(dim=1)[1]
                    id_pred_status.append(pred_status.item())
                    if key_frame:
                        key_frame_status = pred_status.item()
            id_pred_status = np.array(id_pred_status)
            status = np.argmax(np.bincount(id_pred_status))
            data["annotations"][i]["status"] = int(status)
            # data["annotations"][i]["status"] = int(key_frame_status)

            pbar.set_postfix(**{'status': status})
            pbar.update(1)

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



