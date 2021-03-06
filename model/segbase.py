"""Base Model for Semantic Segmentation"""
import torch.nn as nn

from .base_models.resnetv1b import resnet50_v1b, resnet101_v1b, resnet152_v1b, resnet34_v1b
from .base_models.resnext import resnext34
from .base_models.resnest import resnest50, resnest101

class SegBaseModel(nn.Module):
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    """

    def __init__(self, backbone='resnet50', pretrained_base=False, dilated=False, **kwargs):
        super(SegBaseModel, self).__init__()

        if backbone == "resnet34":
            self.backbone = resnet34_v1b(pretrained=pretrained_base, dilated=dilated, **kwargs)
            self.base_channel = [64, 128, 256, 512]
        elif backbone == "resnext34":
            self.backbone = resnext34(dilated=dilated, **kwargs)
            self.base_channel = [64, 128, 256, 512]
        elif backbone == 'resnet50':
            self.backbone = resnet50_v1b(pretrained=pretrained_base, dilated=dilated, **kwargs)
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'resnet101':
            self.backbone = resnet101_v1b(pretrained=pretrained_base, dilated=dilated, **kwargs)
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'resnet152':
            self.backbone = resnet152_v1b(pretrained=pretrained_base, dilated=dilated, **kwargs)
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'resnest50':
            self.backbone = resnest50(pretrained=pretrained_base, dilated=dilated)
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'resnest101':
            self.backbone = resnest101(pretrained=pretrained_base, dilated=dilated)
            self.base_channel = [256, 512, 1024, 2048]

        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))


    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        c1 = self.backbone.layer1(x)
        c2 = self.backbone.layer2(c1)
        c3 = self.backbone.layer3(c2)
        c4 = self.backbone.layer4(c3)

        return c1, c2, c3, c4

