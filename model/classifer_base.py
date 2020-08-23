"""Base Model for Semantic Segmentation"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from model.model_utils import init_weights
from model.base_models.resnet import resnet34, resnet50
from model.base_models.resnest import resnest50, resnest101, resnest200, resnest269
from model.base_models.PyConvResNet.pyconvresnet import pyconvresnet50, pyconvresnet101
from model.base_models.EfficientNet.model import EfficientNet
from model.base_models.densenet import densenet121, densenet169, densenet201
from model.deeplabv3_plus import DeepLabV3Plus


class deep_conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        inter_channel = 16
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, inter_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(True),
            nn.Conv2d(inter_channel, inter_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(True),
            nn.Conv2d(inter_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        return self.conv(x)


class conv_bn_relu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, dilation=1, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.conv(x)


class classifer_base(nn.Module):

    def __init__(self, backbone='resnet50', pretrained_base=True, n_class=2, in_channel=3, **kwargs):
        super(classifer_base, self).__init__()
        self.in_channel = in_channel
        self.net = backbone
        if pretrained_base:
            self.n_class = 1000
        else:
            self.n_class = n_class
        #
        # self.mask_conv2 = deep_conv(1, 32)
        # self.conv_f = conv_bn_relu(96, 64)

        if backbone == "resnet34":
            self.backbone = resnet34(pretrained=pretrained_base, num_classes=self.n_class, in_channel=in_channel, **kwargs)
            self.base_channel = [64, 128, 256, 512]
        elif backbone == 'resnet50':
            self.backbone = resnet50(pretrained=pretrained_base, num_classes=self.n_class, in_channel=in_channel, **kwargs)
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'resnest50':
            self.backbone = resnest50(pretrained=pretrained_base,  num_classes=self.n_class, in_channel=in_channel, **kwargs)
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'resnest101':
            self.backbone = resnest101(pretrained=pretrained_base,  num_classes=self.n_class, in_channel=in_channel, **kwargs)
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'resnest200':
            self.backbone = resnest200(pretrained=pretrained_base,  num_classes=self.n_class, in_channel=in_channel, **kwargs)
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'resnest269':
            self.backbone = resnest269(pretrained=pretrained_base,  num_classes=self.n_class, in_channel=in_channel, **kwargs)
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'pyconvresnet50':
            self.backbone = pyconvresnet50(pretrained=pretrained_base,  num_classes=self.n_class, in_channel=in_channel, **kwargs)
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'pyconvresnet101':
            self.backbone = pyconvresnet101(pretrained=pretrained_base,  num_classes=self.n_class, in_channel=in_channel, **kwargs)
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'DeepLabV3Plus':
            self.backbone = DeepLabV3Plus(2, pretrained_base=pretrained_base)
            self.base_channel = [256]
        elif "efficientnet" in backbone:
            if pretrained_base:
                self.backbone = EfficientNet.from_pretrained(backbone)
            else:
                self.backbone = EfficientNet.from_name(backbone, in_channels=in_channel, num_classes=n_class)
            self.base_channel = [self.backbone.out_channels]
        elif backbone == 'densenet121':
            self.backbone = densenet121(pretrained=pretrained_base, num_classes=self.n_class)
            self.base_channel = [self.backbone.num_features]
        elif backbone == 'densenet169':
            self.backbone = densenet169(pretrained=pretrained_base, num_classes=self.n_class)
            self.base_channel = [self.backbone.num_features]
        elif backbone == 'densenet201':
            self.backbone = densenet201(pretrained=pretrained_base, num_classes=self.n_class)
            self.base_channel = [self.backbone.num_features]
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        # self.CMSI = SPSP(self.base_channel[-1], scales=[13, 6, 3, 1])
        # self.aspp = _ASPP(self.base_channel[-1], [4, 8, 12], norm_layer=nn.BatchNorm2d, norm_kwargs=None, out_channels=512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(self.base_channel[-1], n_class)
        # self.relu = nn.ReLU(True)
        # self.fc2 = nn.Linear(self.base_channel[-1], n_class)

    def forward(self, x):

        x = self.extract_features(x)
        x = self.drop(x)
        x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)

        return x


    def extract_features(self, x):
        # x1 = self.backbone.conv1(x[:, 0:3, :, :])  # 原图
        # x1 = self.backbone.bn1(x1)
        # x1 = self.backbone.relu(x1)
        # x2 = self.mask_conv2(x[:, 3, :, :].unsqueeze(1))  # mask
        # x = self.conv_f(torch.cat((x1, x2), dim=1))

        # x = x[:, 0:3, :, :]

        if self.net == 'DeepLabV3Plus':
            x = self.backbone(x)
        elif "efficientnet" in self.net:
            x = self.backbone.extract_features(x)
        elif "densenet" in self.net:
            x = self.backbone.features(x)
            x = F.relu(x, True)
        else:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x