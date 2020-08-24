
from dataset.dataset_single import AMAP_Dataset_Single, AMAP_PredictDataset
import torch
from model.deeplabv3_plus import DeepLabV3Plus

if __name__ == "__main__":
    data_root = '/home/sjh/dataset/AMAP-TECH/amap_traffic_train_0712'
    target_root = '/home/sjh/dataset/AMAP-TECH/amap_traffic_annotations_train.json'
    crop_size = (360, 640)   # (h, w)
    mode = "test"
    dataset = AMAP_Dataset_Single(data_root, target_root, crop_size, mode)

    image, label, id = dataset[0]
    from torchvision.utils import save_image
    save_image(image, "image.png")
    print(image.size())
    print(label)
    print(id+1)



