# -*- coding: utf-8 -*-
# @Time    : 2019/6/20 16:19
# @Author  : SamChen
# @File    : base.py

import torch.nn as nn

from encoding.models import DRNet_pytorch as dilated_resnet
from encoding.models import ResNet_V2 as ResNet_v2
from encoding.models import ResNext as resnext
from encoding.models import Xception as xception

__all__ = ['BaseNet']

upsample_kwarges = {'mode': 'bilinear', 'align_corners': True}


class BaseNet(nn.Module):
    def __init__(self, n_class, backbone, aux, se_loss, dilated=True, batchnorm=None, pretrained=True,
                 img_size=(192, 256), mean=[.485, .456, .406], std=[.229, .224, .225], root='./pretrain_models',
                 **kwargs):
        super(BaseNet, self).__init__()
        self.n_class = n_class
        self.aux = aux
        self.se_loss = se_loss
        self.mean = mean
        self.std = std
        self.img_size = img_size
        self.backbone = backbone
        print(kwargs)

        if backbone == 'resnet50':
            self.pretrain_model = dilated_resnet.resnet50(pretrained, dilated=dilated, batchnorm=batchnorm, root=root,
                                                          **kwargs)
        elif backbone == 'resnet101':
            self.pretrain_model = dilated_resnet.resnet101(pretrained, dilated=dilated, batchnorm=batchnorm, root=root,
                                                           **kwargs)
        elif backbone == 'resnet101_v3':
            self.pretrain_model = dilated_resnet.resnet101_v3(pretrained, dilated=dilated, batchnorm=batchnorm, root=root, **kwargs)
        elif backbone == 'resnet152':
            self.pretrain_model = dilated_resnet.resnet152(pretrained, dilated=dilated, batchnorm=batchnorm, root=root,
                                                           **kwargs)
        elif backbone == 'resnet50_v2':
            self.pretrain_model = ResNet_v2.resnet50(pretrained, dilated=dilated, batchnorm=batchnorm, root=root,
                                                     **kwargs)
        elif backbone == 'resnet101_v2':
            self.pretrain_model = ResNet_v2.resnet101(pretrained, dilated=dilated, batchnorm=batchnorm, root=root,
                                                      **kwargs)
        elif backbone == 'resnet152_v2':
            self.pretrain_model = ResNet_v2.resnet152(pretrained, dilated=dilated, batchnorm=batchnorm, root=root,
                                                      **kwargs)
        elif backbone == 'resnext50':
            self.pretrain_model = resnext.resnext50(pretrained, dilated=dilated, root=root, **kwargs)
        elif backbone == 'resnext101':
            self.pretrain_model = resnext.resnext101(pretrained, dilated=dilated, root=root, **kwargs)
        elif backbone == 'resnext152':
            self.pretrain_model = resnext.resnext152(pretrained, dilated=dilated, root=root, **kwargs)
        elif backbone == 'xception65':
            self.pretrain_model = xception.get_xception65(pretrained, root=root)
        elif backbone == 'xception71':
            self.pretrain_model = xception.get_xception_71(pretrained, root=root)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        # bilinear upsample options
        self._up_kwargs = upsample_kwarges

    def base_forward(self, x):
        if self.backbone.__contains__('xception'):
            x = self.pretrain_model.conv1(x)
            x = self.pretrain_model.bn1(x)
            x = self.pretrain_model.relu(x)

            x = self.pretrain_model.conv2(x)
            x = self.pretrain_model.bn2(x)
            x = self.pretrain_model.relu(x)

            x = self.pretrain_model.block1(x)
            x = self.pretrain_model.relu(x)
            c1 = x

            x = self.pretrain_model.block2(x)
            x = self.pretrain_model.block3(x)
            c2 = x

            x = self.pretrain_model.midflow(x)
            c3 = x

            x = self.pretrain_model.block20(x)
            x = self.pretrain_model.relu(x)
            x = self.pretrain_model.conv3(x)
            x = self.pretrain_model.bn3(x)
            x = self.pretrain_model.relu(x)

            x = self.pretrain_model.conv4(x)
            x = self.pretrain_model.bn4(x)
            x = self.pretrain_model.relu(x)

            x = self.pretrain_model.conv5(x)
            x = self.pretrain_model.bn5(x)
            x = self.pretrain_model.relu(x)
            c4 = x

        else:
            x = self.pretrain_model.conv1(x)
            x = self.pretrain_model.bn1(x)
            x = self.pretrain_model.relu(x)
            x = self.pretrain_model.maxpool(x)
            c1 = self.pretrain_model.layer1(x)
            c2 = self.pretrain_model.layer2(c1)
            c3 = self.pretrain_model.layer3(c2)
            c4 = self.pretrain_model.layer4(c3)
        # print('c4: {}'.format(c4.size()))

        return c1, c2, c3, c4
