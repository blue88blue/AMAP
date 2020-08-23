from functools import partial
import math
from model.model_utils import init_weights, _FCNHead
import numpy as np
from model.segbase import SegBaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import *
from ..PSPNet import _PyramidPooling
from ..DeepLabV3 import _ASPP
from .SPUnet import SPSP
from .RecoNet import Reco_module

class EMAU(nn.Module):
    '''The Expectation-Maximization Attention Unit (EMAU).

    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''

    def __init__(self, c, k, stage_num=3):
        super(EMAU, self).__init__()
        self.stage_num = stage_num

        # 初始化基
        mu = torch.Tensor(1, c, k)  # k个描述子
        mu.normal_(0, math.sqrt(2. / k))  # Init with Kaiming Norm.
        mu = self._l2norm(mu, dim=1)  # 归一化
        self.register_buffer('mu', mu)

        self.conv1 = nn.Conv2d(c, c, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        idn = x
        # The first 1x1 conv
        x = self.conv1(x)

        # The EM Attention
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)  # b * c * n
        # 批中的每个图片都复制一个基
        mu = self.mu.repeat(b, 1, 1)  # b * c * k
        with torch.no_grad():
            for i in range(self.stage_num):
                x_t = x.permute(0, 2, 1)  # b * n * c
                z = torch.bmm(x_t, mu)  # b * n * k  # 特征图中的n个点与k个基的相似性
                z = F.softmax(z, dim=2)  # b * n * k  # 每个点属于某一个基的概率
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))  # 计算每个点对基的归一化的权重，
                mu = torch.bmm(x, z_)  # b * c * k  # 用每个点去加权组合，得到基
                mu = self._l2norm(mu, dim=1)

        # !!! The moving averaging operation is writtern in train.py, which is significant.

        z_t = z.permute(0, 2, 1)  # b * k * n
        x = mu.matmul(z_t)  # b * c * n  # 用基重建特征图
        x = x.view(b, c, h, w)  # b * c * h * w
        x = F.relu(x, inplace=True)

        # The second 1x1 conv
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x, inplace=True)

        return x, mu

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.

        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.

        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.

        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))





class deep_conv(nn.Module):
    def __init__(self, in_channel, inter_channel,  out_channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, inter_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(True),
            nn.Conv2d(inter_channel, inter_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(True),
            nn.Conv2d(inter_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        return self.conv(x)


class out_conv(nn.Module):
    def __init__(self, in_channel, n_class):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, n_class, kernel_size=1, bias=False),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        return self.conv(x)


class EMANet(SegBaseModel):

    def __init__(self, n_class, backbone='resnet34', aux=False, pretrained_base=False, dilated=True, deep_stem=False,
                 crop_size=224,
                 **kwargs):
        super(EMANet, self).__init__(backbone, pretrained_base=pretrained_base, dilated=dilated,
                                     deep_stem=deep_stem, **kwargs)
        self.aux = aux
        self.dilated = dilated
        channels = self.base_channel  # [256, 512, 1024, 2048]
        if deep_stem or backbone == 'resnest101':
            conv1_channel = 128
        else:
            conv1_channel = 64

        # self.scale = [(128, 128), (256, 256)] # locate
        # self.scale = [(64, 64), (128, 128)]  # seg 224
        # self.conv2 = deep_conv(3, conv1_channel//2, conv1_channel)
        # self.conv3 = deep_conv(3, conv1_channel//2, conv1_channel)

        # self.aspp = _ASPP(channels[3], [6, 10, 14], norm_layer=nn.BatchNorm2d, norm_kwargs=None, out_channels=512, **kwargs)
        self.emau = EMAU(channels[3], k=64)
        # self.ppm = _PyramidPooling(channels[3], norm_layer=nn.BatchNorm2d, norm_kwargs=None)
        # self.conv_ppm = conv_bn_relu(channels[3] * 2, channels[2])

        if dilated:
            self.donv_up3 = decoder_block(channels[0] + channels[3], channels[0])
            self.donv_up4 = decoder_block(channels[0] + conv1_channel, channels[0])
            self.out_conv = out_conv(channels[0]*3, n_class)
        else:
            self.donv_up1 = decoder_block(channels[2] + channels[3], channels[2])
            self.donv_up2 = decoder_block(channels[1] + channels[2], channels[1])
            self.donv_up3 = decoder_block(channels[0] + channels[1], channels[0])
            self.donv_up4 = decoder_block(channels[0] + conv1_channel, channels[0])
            self.out_conv1 = conv_bn_relu(channels[2], channels[0] // 2)
            self.out_conv2 = conv_bn_relu(channels[1], channels[0] // 2)
            self.out_conv3 = conv_bn_relu(channels[0], channels[0] // 2)
            self.out_conv = out_conv(channels[0] * 2 + channels[0] // 2, n_class)

        if type(crop_size) == tuple:
            self.reco = Reco_module(channels[0], crop_size[0] // 2, crop_size[1] // 2, 64)
        else:
            self.reco = Reco_module(channels[0], crop_size//2, crop_size//2, 64)

        if self.aux:
            self.aux_layer = _FCNHead(channels[3], n_class)

        # self.out_conv = out_conv(channels[0], n_class)

    def forward(self, x):
        outputs = []
        size = x.size()[2:]
        # x2 = F.interpolate(x, size=self.scale[0], mode="bilinear", align_corners=True)
        # x3 = F.interpolate(x, size=self.scale[1], mode="bilinear", align_corners=True)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        # x2 = F.interpolate(self.conv2(x2), size=x.size()[2:], mode="bilinear", align_corners=True)
        # x3 = F.interpolate(self.conv3(x3), size=x.size()[2:], mode="bilinear", align_corners=True)
        # x = x + x2 + x3

        c1 = self.backbone.relu(x)  # 1/2  64
        x = self.backbone.maxpool(c1)
        c2 = self.backbone.layer1(x)  # 1/4   64
        c3 = self.backbone.layer2(c2)  # 1/8   128
        c4 = self.backbone.layer3(c3)  # 1/16   256
        c5 = self.backbone.layer4(c4)  # 1/32   512

        # c5 = self.aspp(c5)
        c5, mu = self.emau(c5)
        # c5 = self.conv_ppm(self.ppm(c5))

        if self.dilated:
            x = self.donv_up3(c5, c2)
            x = self.donv_up4(x, c1)
        else:
            x1 = self.donv_up1(c5, c4)
            x2 = self.donv_up2(x1, c3)
            x3 = self.donv_up3(x2, c2)
            x4 = self.donv_up4(x3, c1)

            x1 = F.interpolate(self.out_conv1(x1), x4.size()[-2:], mode='bilinear', align_corners=True)  # 最后上采样
            x2 = F.interpolate(self.out_conv2(x2), x4.size()[-2:], mode='bilinear', align_corners=True)  # 最后上采样
            x3 = F.interpolate(self.out_conv3(x3), x4.size()[-2:], mode='bilinear', align_corners=True)  # 最后上采样
            x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.reco(x)
        x = self.out_conv(x)

        x = F.interpolate(x, size, mode='bilinear', align_corners=True)  # 最后上采样
        outputs.append(x)
        outputs.append(mu)

        if self.aux:
            auxout = self.aux_layer(c5)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return outputs
