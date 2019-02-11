from .resnet import *
from .deeplabv2 import *
from .msc import MSC


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def DeepLabV2_ResNet101_MSC(n_classes):
    return MSC(
        base=DeepLabV2(
            n_classes=n_classes, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
        ),
        scales=[0.5, 0.75],
    )


def DeepLabV2S_ResNet101_MSC(n_classes):
    return MSC(
        base=DeepLabV2(
            n_classes=n_classes, n_blocks=[3, 4, 23, 3], atrous_rates=[3, 6, 9, 12]
        ),
        scales=[0.5, 0.75],
    )
