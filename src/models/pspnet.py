"""Pyramid Scene Parsing Network"""
from gluoncv.model_zoo.fcn import _FCNHead
from gluoncv.model_zoo.segbase import SegBaseModel
from mxnet.context import cpu
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock


class PSPNet(SegBaseModel):
    r"""Pyramid Scene Parsing Network

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.


    Reference:

        Zhao, Hengshuang, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia.
        "Pyramid scene parsing network." *CVPR*, 2017

    """

    def __init__(self, nclass, backbone='resnet50', aux=True, ctx=cpu(), pretrained_base=True,
                 base_size=520, crop_size=480, **kwargs):
        super(PSPNet, self).__init__(nclass, aux, backbone, ctx=ctx, base_size=base_size,
                                     crop_size=crop_size, pretrained_base=pretrained_base, **kwargs)
        with self.name_scope():
            self.head = _PSPHead(nclass, **kwargs)
            self.head.initialize(ctx=ctx)
            self.head.collect_params().setattr('lr_mult', 10)
            if self.aux:
                self.auxlayer = _FCNHead(1024, nclass, **kwargs)
                self.auxlayer.initialize(ctx=ctx)
                self.auxlayer.collect_params().setattr('lr_mult', 10)
        print('self.crop_size', self.crop_size)

    def hybrid_forward(self, F, x):
        c3, c4 = self.base_forward(x)
        outputs = []
        x = self.head(c4)
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.contrib.BilinearResize2D(auxout, **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)


def _PSP1x1Conv(in_channels, out_channels, norm_layer, norm_kwargs):
    block = nn.HybridSequential()
    with block.name_scope():
        block.add(nn.Conv2D(in_channels=in_channels, channels=out_channels,
                            kernel_size=1, use_bias=False))
        block.add(norm_layer(in_channels=out_channels, **({} if norm_kwargs is None else norm_kwargs)))
        block.add(nn.Activation('relu'))
    return block


class _PyramidPooling(HybridBlock):
    def __init__(self, in_channels, **kwargs):
        super(_PyramidPooling, self).__init__()
        out_channels = int(in_channels / 4)
        with self.name_scope():
            self.conv1 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
            self.conv2 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
            self.conv3 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
            self.conv4 = _PSP1x1Conv(in_channels, out_channels, **kwargs)

    def pool(self, F, x, size):
        return F.contrib.AdaptiveAvgPooling2D(x, output_size=size)

    def upsample(self, F, x, h, w):
        return F.contrib.BilinearResize2D(x, height=h, width=w)

    def hybrid_forward(self, F, x):
        _, _, h, w = x.shape
        feat1 = self.upsample(F, self.conv1(self.pool(F, x, 1)), h, w)
        feat2 = self.upsample(F, self.conv2(self.pool(F, x, 2)), h, w)
        feat3 = self.upsample(F, self.conv3(self.pool(F, x, 3)), h, w)
        feat4 = self.upsample(F, self.conv4(self.pool(F, x, 4)), h, w)
        return F.concat(x, feat1, feat2, feat3, feat4, dim=1)


class _PSPHead(HybridBlock):
    def __init__(self, nclass, norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(_PSPHead, self).__init__()
        self.psp = _PyramidPooling(2048, norm_layer=norm_layer,
                                   norm_kwargs=norm_kwargs)
        with self.name_scope():
            self.block = nn.HybridSequential(prefix='')
            self.block.add(nn.Conv2D(in_channels=4096, channels=512,
                                     kernel_size=3, padding=1, use_bias=False))
            self.block.add(norm_layer(in_channels=512, **({} if norm_kwargs is None else norm_kwargs)))
            self.block.add(nn.Activation('relu'))
            self.block.add(nn.Dropout(0.1))
            self.block.add(nn.Conv2D(in_channels=512, channels=nclass,
                                     kernel_size=1))

    def hybrid_forward(self, F, x):
        x = self.psp(x)
        return self.block(x)
