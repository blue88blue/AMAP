from model.segbase import SegBaseModel
from model.model_utils import init_weights, _FCNHead
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .blocks import *
from .shuffle import myInvertedResidual, fusion
from .SPUnet import SPSP
from ..DeepLabV3 import _ASPP
from torchvision.utils import save_image
import time

class ResUnet(SegBaseModel):

    def __init__(self, n_class, backbone='resnet34', aux=False, pretrained_base=False, dilated=True, deep_stem=False, **kwargs):
        super(ResUnet, self).__init__(backbone, pretrained_base=pretrained_base, dilated=dilated, deep_stem=deep_stem, **kwargs)
        self.aux = aux
        self.dilated = dilated
        channels = self.base_channel
        if deep_stem or backbone == 'resnest101':
            conv1_channel = 128
        else:
            conv1_channel = 64

        self.spsp = SPSP(channels[3], scales=[14, 7, 3, 1])

        if dilated:
            self.donv_up3 = decoder_block(channels[0]+channels[3], channels[0])
            self.donv_up4 = decoder_block(channels[0]+conv1_channel, channels[0])
        else:
            self.donv_up1 = decoder_block(channels[2] + channels[3], channels[2])
            self.donv_up2 = decoder_block(channels[1] + channels[2], channels[1])
            self.donv_up3 = decoder_block(channels[0] + channels[1], channels[0])
            self.donv_up4 = decoder_block(channels[0] + conv1_channel, channels[0])

        if self.aux:
            self.aux_layer = _FCNHead(256, n_class)

        self.out_conv = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
            nn.Conv2d(channels[0], n_class, kernel_size=1, bias=False),
        )

    def forward(self, x):
        outputs = []
        size = x.size()[2:]
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        c1 = self.backbone.relu(x)  # 1/2  64
        x = self.backbone.maxpool(c1)
        c2 = self.backbone.layer1(x)  # 1/4   64
        c3 = self.backbone.layer2(c2)  # 1/8   128
        c4 = self.backbone.layer3(c3)  # 1/16   256
        c5 = self.backbone.layer4(c4)  # 1/32   512

        c5 = self.spsp(c5)

        if self.dilated:
            x = self.donv_up3(c5, c2)
            x = self.donv_up4(x, c1)
        else:
            x = self.donv_up1(c5, c4)
            x = self.donv_up2(x, c3)
            x = self.donv_up3(x, c2)
            x = self.donv_up4(x, c1)

        x = self.out_conv(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)  # 最后上采样

        outputs.append(x)
        if self.aux:
            auxout = self.aux_layer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return outputs






class shuffle_ResUnet(SegBaseModel):

    def __init__(self, n_class, backbone='resnet34', aux=False, pretrained_base=False, dilated=False, deep_stem=False, **kwargs):
        super(shuffle_ResUnet, self).__init__(backbone, pretrained_base=pretrained_base, dilated=dilated, deep_stem=deep_stem, **kwargs)
        self.aux = aux
        channels = [64, 128, 256, 512]
        if deep_stem:
            conv1_channel = 128
        else:
            conv1_channel = 64

        self.shuconv_mid1 = myInvertedResidual(channels[3], [1, 2, 4])
        self.shuconv_mid2 = myInvertedResidual(channels[3], [1, 2, 4])
        self.shuconv_mid3 = myInvertedResidual(channels[3], [1, 2, 4])

        self.donv_up1 = fusion(channels[3], channels[2], channels[2])
        self.donv_up2 = fusion(channels[2], channels[1], channels[1])
        self.donv_up3 = fusion(channels[1], channels[0], channels[0])
        self.donv_up4 = fusion(channels[0], conv1_channel, channels[0])

        self.out_conv = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
            nn.Conv2d(channels[0], n_class, kernel_size=1, bias=False),
        )

    def forward(self, x):
        outputs = []
        size = x.size()[2:]
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        c1 = self.backbone.relu(x)  # 1/2  64
        x = self.backbone.maxpool(c1)
        c2 = self.backbone.layer1(x)  # 1/4   64
        c3 = self.backbone.layer2(c2)  # 1/8   128
        c4 = self.backbone.layer3(c3)  # 1/16   256
        c5 = self.backbone.layer4(c4)  # 1/32   512

        c5 = self.shuconv_mid1(c5)
        c5 = self.shuconv_mid2(c5)
        c5 = self.shuconv_mid3(c5)

        x = self.donv_up1(c5, c4)
        x = self.donv_up2(x, c3)
        x = self.donv_up3(x, c2)
        x = self.donv_up4(x, c1)

        x = F.interpolate(x, size, mode='bilinear', align_corners=True)  # 最后上采样
        x = self.out_conv(x)

        outputs.append(x)
        return outputs



