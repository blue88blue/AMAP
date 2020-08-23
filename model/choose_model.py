from model.Unet import Unet
from model.AttUnet import AttUnet
from model.PSPNet import PSPNet
from model.DeepLabV3 import DeepLabV3
from model.DANet import DANet
from model.CPFNet import CPFNet
from model.BiFPN import ResNet_BiFPN
from model.AGNet.model import AG_Net
from model.my_model.ResUnet import ResUnet, shuffle_ResUnet
from model.cenet import CE_Net_
from model.my_model.shuffle import shuffle_Unet, shuffle_fusion_Unet
from model.my_model.SPUnet import SP_Unet
from model.my_model.EMANet import EMANet
from model.my_model.multi_ResUnet import Multi_ResUnet
from model.deeplabv3_plus import DeepLabV3Plus


def seg_model(args):
    if args.network == "Unet":
        model = Unet(args.in_channel, args.n_class, channel_reduction=args.Ulikenet_channel_reduction, aux=args.aux)
    elif args.network == "AttUnet":
        model = AttUnet(args.in_channel, args.n_class, channel_reduction=args.Ulikenet_channel_reduction, aux=args.aux)
    elif args.network == "PSPNet":
        model = PSPNet(args.n_class, args.backbone, aux=args.aux, pretrained_base=args.pretrained, dilated=args.dilated, deep_stem=args.deep_stem)
    elif args.network == "DeepLabV3":
        model = DeepLabV3(args.n_class, args.backbone, aux=args.aux, pretrained_base=args.pretrained, dilated=args.dilated, deep_stem=args.deep_stem)
    elif args.network == "DANet":
        model = DANet(args.n_class, args.backbone, aux=args.aux, pretrained_base=args.pretrained, dilated=args.dilated, deep_stem=args.deep_stem)
    elif args.network == "CPFNet":
        model = CPFNet(args.n_class, args.backbone, aux=args.aux, pretrained_base=args.pretrained, dilated=args.dilated, deep_stem=args.deep_stem)
    elif args.network == "AG_Net":
        model = AG_Net(args.n_class)
    elif args.network == "CENet":
        model = CE_Net_(args.n_class)
    elif args.network == "ResUnet":
        model = ResUnet(args.n_class, args.backbone, aux=args.aux, pretrained_base=args.pretrained, dilated=args.dilated, deep_stem=args.deep_stem)

    elif args.network == "shuffle_Unet":
        model = shuffle_Unet(args.in_channel, args.n_class, aux=args.aux)
    elif args.network == "shuffle_fusion_Unet":
        model = shuffle_fusion_Unet(args.in_channel, args.n_class, aux=args.aux)
    elif args.network == "shuffle_ResUnet":
        model = shuffle_ResUnet(args.n_class, args.backbone, aux=args.aux, pretrained_base=args.pretrained, dilated=args.dilated, deep_stem=args.deep_stem)
    elif args.network == "SP_Unet":
        model = SP_Unet(args.in_channel, args.n_class, aux=args.aux)

    elif args.network == "EMANet":
        model = EMANet(args.n_class, args.backbone, aux=args.aux, pretrained_base=args.pretrained, dilated=args.dilated, deep_stem=args.deep_stem, crop_size=args.crop_size)
    elif args.network == "Multi_ResUnet":
        model = Multi_ResUnet(args.n_class, args.backbone, aux=args.aux, pretrained_base=args.pretrained, dilated=args.dilated, deep_stem=args.deep_stem)
    elif args.network == "DeepLabV3Plus":
        model = DeepLabV3Plus(args.n_class)
    else:
        NotImplementedError("not implemented {args.network} model")

    return model














