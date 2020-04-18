# -*- coding: utf-8 -*-
# @Time    : 2019/12/13 17:43
# @Author  : SamChen
# @File    : MultiScaleContrastModel.py


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from ..models.base import BaseNet
from .ASPP import *
from .customize import PyramidPooling

__all__ = ['MultiScaleContrastModel', 'get_exp1_mscm', 'get_exp2_densemscm', 'get_exp2_mscm_v2',
           'get_exp3_mscm_v2_wo_edge', 'get_pspnet']


class channel_contrast_attention(nn.Module):
    def __init__(self, in_channels, out_channals):
        super(channel_contrast_attention, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channals, 1, 1, bias=False)
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.conv_down = nn.Conv2d(out_channals, out_channals // 16, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU(True)
        self.conv_up = nn.Conv2d(out_channals // 16, out_channals, kernel_size=1, stride=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.global_avg(x)
        x = self.conv_down(x)
        x = self.relu(x)
        x = self.conv_up(x)
        attention = self.sigmoid(x)

        return attention


class spacial_contrast_attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(spacial_contrast_attention, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, 1, bias=False)
        self.fc1 = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.relu = nn.ReLU(True)
        self.fc2 = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1x1(x)
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        attention = self.sigmoid(x)

        return attention


class pyramid_contrast_extraction(nn.Module):
    def __init__(self, in_channels, pyramid_kernel=[3, 5]):
        super(pyramid_contrast_extraction, self).__init__()
        self.pooling1 = nn.AvgPool2d(pyramid_kernel[0], stride=1, padding=(pyramid_kernel[0] - 1) // 2)
        self.pooling2 = nn.AvgPool2d(pyramid_kernel[1], stride=1, padding=(pyramid_kernel[1] - 1) // 2)
        self.channel_contrast_att = channel_contrast_attention(in_channels * 2, in_channels)
        self.spacial_contrast_att = spacial_contrast_attention(in_channels * 2, in_channels)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        pooling_feature1 = x - self.pooling1(x)
        pooling_feature2 = x - self.pooling2(x)
        multi_pooling_features = torch.cat([pooling_feature1, pooling_feature2], dim=1)
        channel_attention = self.channel_contrast_att(multi_pooling_features)
        spacial_attention = self.spacial_contrast_att(multi_pooling_features)

        out = x + self.alpha * channel_attention * x + self.beta * spacial_attention * x

        return out


class contrast_extraction(nn.Module):
    def __init__(self, in_channels, pyramid_kernel=3):
        super(contrast_extraction, self).__init__()
        self.pooling1 = nn.AvgPool2d(pyramid_kernel, stride=1, padding=(pyramid_kernel - 1) // 2)
        # self.pooling2 = nn.AvgPool2d(pyramid_kernel[1], stride=1, padding=(pyramid_kernel[1] - 1) // 2)
        self.channel_contrast_att = channel_contrast_attention(in_channels * 2, in_channels)
        self.spacial_contrast_att = spacial_contrast_attention(in_channels * 2, in_channels)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        pooling_feature1 = x - self.pooling1(x)
        # pooling_feature2 = x - self.pooling2(x)
        multi_pooling_features = torch.cat([x, pooling_feature1], dim=1)
        channel_attention = self.channel_contrast_att(multi_pooling_features)
        spacial_attention = self.spacial_contrast_att(multi_pooling_features)

        out = x + self.alpha * channel_attention * x + self.beta * spacial_attention * x

        return out


class contrast_extraction_wo_edge(nn.Module):
    def __init__(self, in_channels, pyramid_kernel=3):
        super(contrast_extraction_wo_edge, self).__init__()
        # self.pooling1 = nn.AvgPool2d(pyramid_kernel, stride=1, padding=(pyramid_kernel - 1) // 2)
        # self.pooling2 = nn.AvgPool2d(pyramid_kernel[1], stride=1, padding=(pyramid_kernel[1] - 1) // 2)
        self.channel_contrast_att = channel_contrast_attention(in_channels, in_channels)
        self.spacial_contrast_att = spacial_contrast_attention(in_channels, in_channels)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # pooling_feature1 = x - self.pooling1(x)
        # pooling_feature2 = x - self.pooling2(x)
        # multi_pooling_features = torch.cat([x, pooling_feature1], dim=1)
        channel_attention = self.channel_contrast_att(x)
        spacial_attention = self.spacial_contrast_att(x)

        out = x + self.alpha * channel_attention * x + self.beta * spacial_attention * x

        return out


class pyramid_contrast_extraction_v2(nn.Module):
    def __init__(self, in_channels, pyramid_kernel=[3, 5]):
        super(pyramid_contrast_extraction_v2, self).__init__()
        self.pooling1 = nn.AvgPool2d(pyramid_kernel[0], stride=1, padding=(pyramid_kernel[0] - 1) // 2)
        # self.pooling2 = nn.AvgPool2d(pyramid_kernel[1], stride=1, padding=(pyramid_kernel[1] - 1) // 2)
        self.channel_contrast_att = channel_contrast_attention(in_channels, in_channels)
        self.spacial_contrast_att = spacial_contrast_attention(in_channels, in_channels)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        pooling_feature1 = x - self.pooling1(x)
        # pooling_feature2 = x - self.pooling2(x)
        # multi_pooling_features = torch.cat([pooling_feature1, pooling_feature2], dim=1)
        channel_attention = self.channel_contrast_att(pooling_feature1)
        spacial_attention = self.spacial_contrast_att(pooling_feature1)

        out = x + self.alpha * channel_attention * x + self.beta * spacial_attention * x

        return out


class features_upsample_fusion(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, batchnorm):
        super(features_upsample_fusion, self).__init__()
        self.conv3x3 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                                     batchnorm(out_channels),
                                     nn.ReLU(True))
        self.conv1x1_1 = nn.Conv2d(in_channels1, out_channels, 1, bias=False)
        self.conv1x1_2 = nn.Conv2d(in_channels2, out_channels, 1, bias=False)

    def forward(self, previous_x, current_x):
        _, _, h, w = current_x.size()
        if previous_x.size()[2:] != (h, w):
            previous_x = F.interpolate(previous_x, (h, w), mode='bilinear', align_corners=True)
        previous_x = self.conv1x1_1(previous_x)
        current_x = self.conv1x1_2(current_x)

        out = self.conv3x3(previous_x + current_x)

        return out


class MultiScaleContrastModel(BaseNet):
    def __init__(self, n_class, backbone, aux=False, se_loss=False, resaux=False, aggaux=False, up_conv=0,
                 batchnorm=nn.BatchNorm2d, activation='relu', is_train=True, aspp_rates=[6, 12, 18], boundary_rf=False,
                 boundary_kernel=5, test_size=[256, 256], ensemble=False, aspp_out_dim=512, agg_out_dim=128,
                 spp_size=None, refine=False, refine_ver='v2', **kwargs):
        super(MultiScaleContrastModel, self).__init__(n_class, backbone, aux, se_loss, batchnorm=batchnorm, **kwargs)
        self.is_trian = is_train
        self.test_size = test_size
        self.ensemble = ensemble
        self.resaux = resaux
        self.aggaux = aggaux
        self.spp = nn.Sequential(PyramidPooling(2048, batchnorm, self._up_kwargs),
                                 nn.Conv2d(4096, agg_out_dim, 3, padding=1, bias=False),
                                 batchnorm(agg_out_dim),
                                 nn.ReLU(True),
                                 )
        self.contrast_extraction1 = pyramid_contrast_extraction(256, pyramid_kernel=[5, 7])
        self.contrast_extraction2 = pyramid_contrast_extraction(512, pyramid_kernel=[5, 7])
        self.contrast_extraction3 = pyramid_contrast_extraction(1024, pyramid_kernel=[3, 5])
        self.contrast_extraction4 = pyramid_contrast_extraction(2048, pyramid_kernel=[3, 5])
        self.upsample_fusion4 = features_upsample_fusion(agg_out_dim, 2048, agg_out_dim, batchnorm)
        self.upsample_fusion3 = features_upsample_fusion(agg_out_dim, 1024, agg_out_dim, batchnorm)
        self.upsample_fusion2 = features_upsample_fusion(agg_out_dim, 512, agg_out_dim, batchnorm)
        self.upsample_fusion1 = features_upsample_fusion(agg_out_dim, 256, agg_out_dim, batchnorm)

        self.final_conv = nn.Sequential(nn.Conv2d(agg_out_dim, agg_out_dim // 4, 3, padding=1, bias=False),
                                        batchnorm(agg_out_dim // 4),
                                        nn.ReLU(True),
                                        nn.Conv2d(agg_out_dim // 4, n_class, 1))

        self.resaux_conv = nn.Sequential(
            nn.Conv2d(1024, 1024 // 4, kernel_size=3, stride=1, padding=1, bias=False),
            batchnorm(1024 // 4),
            nn.ReLU(True),
            # nn.Dropout2d(0.1),
            nn.Conv2d(1024 // 4, n_class, 1))
        self.aggaux_conv = nn.Sequential(
            nn.Conv2d(agg_out_dim, agg_out_dim // 4, kernel_size=3, stride=1, padding=1, bias=False),
            batchnorm(agg_out_dim // 4),
            nn.ReLU(True),
            # nn.Dropout2d(0.1),
            nn.Conv2d(agg_out_dim // 4, n_class, 1))

    def forward(self, x):
        imsize = x.size()[2:]
        if not self.is_trian:
            imsize = self.test_size
        c1, c2, c3, c4 = self.base_forward(x)
        contrast_feats1 = self.contrast_extraction1(c1)
        contrast_feats2 = self.contrast_extraction2(c2)
        contrast_feats3 = self.contrast_extraction3(c3)
        contrast_feats4 = self.contrast_extraction4(c4)

        spp_feats = self.spp(c4)

        d4 = self.upsample_fusion4(spp_feats, contrast_feats4)
        d3 = self.upsample_fusion3(d4, contrast_feats3)
        d2 = self.upsample_fusion2(d3, contrast_feats2)
        d1 = self.upsample_fusion1(d2, contrast_feats1)

        out = self.final_conv(d1)
        out = F.interpolate(out, imsize, mode='bilinear', align_corners=True)

        if self.resaux and not self.aggaux:
            resaux_loss = self.resaux_conv(c3)
            resaux_loss = F.interpolate(resaux_loss, imsize, mode='bilinear', align_corners=True)
            outputs = [out, out, resaux_loss]
        elif not self.resaux and self.aggaux:
            aggaux_loss = self.aggaux_conv(spp_feats)
            aggaux_loss = F.interpolate(aggaux_loss, imsize, mode='bilinear', align_corners=True)
            outputs = [out, out, aggaux_loss]
        elif self.resaux and self.aggaux:
            resaux_loss = self.resaux_conv(c3)
            resaux_loss = F.interpolate(resaux_loss, imsize, mode='bilinear', align_corners=True)
            aggaux_loss = self.aggaux_conv(spp_feats)
            aggaux_loss = F.interpolate(aggaux_loss, imsize, mode='bilinear', align_corners=True)
            outputs = [out, out, resaux_loss, aggaux_loss]
        elif not self.resaux and not self.aggaux:
            outputs = [out, out]

        return tuple(outputs)


class MultiScaleContrastModel_v2(BaseNet):
    def __init__(self, n_class, backbone, aux=False, se_loss=False, resaux=False, aggaux=False, up_conv=0,
                 batchnorm=nn.BatchNorm2d, activation='relu', is_train=True, aspp_rates=[6, 12, 18], boundary_rf=False,
                 boundary_kernel=5, test_size=[256, 256], ensemble=False, aspp_out_dim=512, agg_out_dim=128,
                 spp_size=None, refine=False, refine_ver='v2', **kwargs):
        super(MultiScaleContrastModel_v2, self).__init__(n_class, backbone, aux, se_loss, batchnorm=batchnorm, **kwargs)
        self.is_trian = is_train
        self.test_size = test_size
        self.ensemble = ensemble
        self.resaux = resaux
        self.aggaux = aggaux
        self.spp = nn.Sequential(PyramidPooling(2048, batchnorm, self._up_kwargs),
                                 nn.Conv2d(4096, agg_out_dim, 3, padding=1, bias=False),
                                 batchnorm(agg_out_dim),
                                 nn.ReLU(True),
                                 )
        self.contrast_extraction1 = contrast_extraction(256, pyramid_kernel=7)
        self.contrast_extraction2 = contrast_extraction(512, pyramid_kernel=5)
        self.contrast_extraction3 = contrast_extraction(1024, pyramid_kernel=3)
        self.contrast_extraction4 = contrast_extraction(2048, pyramid_kernel=3)
        self.upsample_fusion4 = features_upsample_fusion(agg_out_dim, 2048, agg_out_dim, batchnorm)
        self.upsample_fusion3 = features_upsample_fusion(agg_out_dim, 1024, agg_out_dim, batchnorm)
        self.upsample_fusion2 = features_upsample_fusion(agg_out_dim, 512, agg_out_dim, batchnorm)
        self.upsample_fusion1 = features_upsample_fusion(agg_out_dim, 256, agg_out_dim, batchnorm)

        self.final_conv = nn.Sequential(nn.Conv2d(agg_out_dim, agg_out_dim // 4, 3, padding=1, bias=False),
                                        batchnorm(agg_out_dim // 4),
                                        nn.ReLU(True),
                                        nn.Conv2d(agg_out_dim // 4, n_class, 1))

        self.resaux_conv = nn.Sequential(
            nn.Conv2d(1024, 1024 // 4, kernel_size=3, stride=1, padding=1, bias=False),
            batchnorm(1024 // 4),
            nn.ReLU(True),
            # nn.Dropout2d(0.1),
            nn.Conv2d(1024 // 4, n_class, 1))
        self.aggaux_conv = nn.Sequential(
            nn.Conv2d(agg_out_dim, agg_out_dim // 4, kernel_size=3, stride=1, padding=1, bias=False),
            batchnorm(agg_out_dim // 4),
            nn.ReLU(True),
            # nn.Dropout2d(0.1),
            nn.Conv2d(agg_out_dim // 4, n_class, 1))

    def forward(self, x):
        imsize = x.size()[2:]
        if not self.is_trian:
            imsize = self.test_size
        c1, c2, c3, c4 = self.base_forward(x)
        contrast_feats1 = self.contrast_extraction1(c1)
        contrast_feats2 = self.contrast_extraction2(c2)
        contrast_feats3 = self.contrast_extraction3(c3)
        contrast_feats4 = self.contrast_extraction4(c4)

        spp_feats = self.spp(c4)

        d4 = self.upsample_fusion4(spp_feats, contrast_feats4)
        d3 = self.upsample_fusion3(d4, contrast_feats3)
        d2 = self.upsample_fusion2(d3, contrast_feats2)
        d1 = self.upsample_fusion1(d2, contrast_feats1)

        out = self.final_conv(d1)
        out = F.interpolate(out, imsize, mode='bilinear', align_corners=True)

        if self.resaux and not self.aggaux:
            resaux_loss = self.resaux_conv(c3)
            resaux_loss = F.interpolate(resaux_loss, imsize, mode='bilinear', align_corners=True)
            outputs = [out, out, resaux_loss]
        elif not self.resaux and self.aggaux:
            aggaux_loss = self.aggaux_conv(spp_feats)
            aggaux_loss = F.interpolate(aggaux_loss, imsize, mode='bilinear', align_corners=True)
            outputs = [out, out, aggaux_loss]
        elif self.resaux and self.aggaux:
            resaux_loss = self.resaux_conv(c3)
            resaux_loss = F.interpolate(resaux_loss, imsize, mode='bilinear', align_corners=True)
            aggaux_loss = self.aggaux_conv(spp_feats)
            aggaux_loss = F.interpolate(aggaux_loss, imsize, mode='bilinear', align_corners=True)
            outputs = [out, out, resaux_loss, aggaux_loss]
        elif not self.resaux and not self.aggaux:
            outputs = [out, out]

        return tuple(outputs)


class MultiScaleContrastModel_wo_edge(BaseNet):
    def __init__(self, n_class, backbone, aux=False, se_loss=False, resaux=False, aggaux=False, up_conv=0,
                 batchnorm=nn.BatchNorm2d, activation='relu', is_train=True, aspp_rates=[6, 12, 18], boundary_rf=False,
                 boundary_kernel=5, test_size=[256, 256], ensemble=False, aspp_out_dim=512, agg_out_dim=128,
                 spp_size=None, refine=False, refine_ver='v2', **kwargs):
        super(MultiScaleContrastModel_wo_edge, self).__init__(n_class, backbone, aux, se_loss, batchnorm=batchnorm,
                                                              **kwargs)
        self.is_trian = is_train
        self.test_size = test_size
        self.ensemble = ensemble
        self.resaux = resaux
        self.aggaux = aggaux
        self.spp = nn.Sequential(PyramidPooling(2048, batchnorm, self._up_kwargs),
                                 nn.Conv2d(4096, agg_out_dim, 3, padding=1, bias=False),
                                 batchnorm(agg_out_dim),
                                 nn.ReLU(True),
                                 )
        self.contrast_extraction1 = contrast_extraction_wo_edge(256, pyramid_kernel=7)
        self.contrast_extraction2 = contrast_extraction_wo_edge(512, pyramid_kernel=5)
        self.contrast_extraction3 = contrast_extraction_wo_edge(1024, pyramid_kernel=3)
        self.contrast_extraction4 = contrast_extraction_wo_edge(2048, pyramid_kernel=3)
        self.upsample_fusion4 = features_upsample_fusion(agg_out_dim, 2048, agg_out_dim, batchnorm)
        self.upsample_fusion3 = features_upsample_fusion(agg_out_dim, 1024, agg_out_dim, batchnorm)
        self.upsample_fusion2 = features_upsample_fusion(agg_out_dim, 512, agg_out_dim, batchnorm)
        self.upsample_fusion1 = features_upsample_fusion(agg_out_dim, 256, agg_out_dim, batchnorm)

        self.final_conv = nn.Sequential(nn.Conv2d(agg_out_dim, agg_out_dim // 4, 3, padding=1, bias=False),
                                        batchnorm(agg_out_dim // 4),
                                        nn.ReLU(True),
                                        nn.Conv2d(agg_out_dim // 4, n_class, 1))

        self.resaux_conv = nn.Sequential(
            nn.Conv2d(1024, 1024 // 4, kernel_size=3, stride=1, padding=1, bias=False),
            batchnorm(1024 // 4),
            nn.ReLU(True),
            # nn.Dropout2d(0.1),
            nn.Conv2d(1024 // 4, n_class, 1))
        self.aggaux_conv = nn.Sequential(
            nn.Conv2d(agg_out_dim, agg_out_dim // 4, kernel_size=3, stride=1, padding=1, bias=False),
            batchnorm(agg_out_dim // 4),
            nn.ReLU(True),
            # nn.Dropout2d(0.1),
            nn.Conv2d(agg_out_dim // 4, n_class, 1))

    def forward(self, x):
        imsize = x.size()[2:]
        if not self.is_trian:
            imsize = self.test_size
        c1, c2, c3, c4 = self.base_forward(x)
        contrast_feats1 = self.contrast_extraction1(c1)
        contrast_feats2 = self.contrast_extraction2(c2)
        contrast_feats3 = self.contrast_extraction3(c3)
        contrast_feats4 = self.contrast_extraction4(c4)

        spp_feats = self.spp(c4)

        d4 = self.upsample_fusion4(spp_feats, contrast_feats4)
        d3 = self.upsample_fusion3(d4, contrast_feats3)
        d2 = self.upsample_fusion2(d3, contrast_feats2)
        d1 = self.upsample_fusion1(d2, contrast_feats1)

        out = self.final_conv(d1)
        out = F.interpolate(out, imsize, mode='bilinear', align_corners=True)

        if self.resaux and not self.aggaux:
            resaux_loss = self.resaux_conv(c3)
            resaux_loss = F.interpolate(resaux_loss, imsize, mode='bilinear', align_corners=True)
            outputs = [out, out, resaux_loss]
        elif not self.resaux and self.aggaux:
            aggaux_loss = self.aggaux_conv(spp_feats)
            aggaux_loss = F.interpolate(aggaux_loss, imsize, mode='bilinear', align_corners=True)
            outputs = [out, out, aggaux_loss]
        elif self.resaux and self.aggaux:
            resaux_loss = self.resaux_conv(c3)
            resaux_loss = F.interpolate(resaux_loss, imsize, mode='bilinear', align_corners=True)
            aggaux_loss = self.aggaux_conv(spp_feats)
            aggaux_loss = F.interpolate(aggaux_loss, imsize, mode='bilinear', align_corners=True)
            outputs = [out, out, resaux_loss, aggaux_loss]
        elif not self.resaux and not self.aggaux:
            outputs = [out, out]

        return tuple(outputs)


class DenseMultiScaleContrastModel(BaseNet):
    def __init__(self, n_class, backbone, aux=False, se_loss=False, resaux=False, aggaux=False, up_conv=0,
                 batchnorm=nn.BatchNorm2d, activation='relu', is_train=True, aspp_rates=[6, 12, 18], boundary_rf=False,
                 boundary_kernel=5, test_size=[256, 256], ensemble=False, aspp_out_dim=512, agg_out_dim=128,
                 spp_size=None, refine=False, refine_ver='v2', **kwargs):
        super(DenseMultiScaleContrastModel, self).__init__(n_class, backbone, aux, se_loss, batchnorm=batchnorm,
                                                           **kwargs)
        self.is_trian = is_train
        self.test_size = test_size
        self.ensemble = ensemble
        self.resaux = resaux
        self.aggaux = aggaux
        self.spp = nn.Sequential(PyramidPooling(2048, batchnorm, self._up_kwargs),
                                 nn.Conv2d(4096, agg_out_dim, 3, padding=1, bias=False),
                                 batchnorm(agg_out_dim),
                                 nn.ReLU(True),
                                 )
        self.contrast_extraction1 = pyramid_contrast_extraction(256, pyramid_kernel=[5, 7])
        self.contrast_extraction2 = pyramid_contrast_extraction(512, pyramid_kernel=[5, 7])
        self.contrast_extraction3 = pyramid_contrast_extraction(1024, pyramid_kernel=[3, 5])
        self.contrast_extraction4 = pyramid_contrast_extraction(2048, pyramid_kernel=[3, 5])
        # self.dense_conv4 = nn.Conv2d(2 * agg_out_dim, agg_out_dim, 3, padding=1, bias=False)
        self.upsample_fusion4 = features_upsample_fusion(agg_out_dim, 2048, agg_out_dim, batchnorm)
        self.dense_conv3 = nn.Conv2d(2 * agg_out_dim, agg_out_dim, 1, bias=False)
        self.upsample_fusion3 = features_upsample_fusion(agg_out_dim, 1024, agg_out_dim, batchnorm)
        self.dense_conv2 = nn.Conv2d(3 * agg_out_dim, agg_out_dim, 1, bias=False)
        self.upsample_fusion2 = features_upsample_fusion(agg_out_dim, 512, agg_out_dim, batchnorm)
        self.upsample_fusion1 = features_upsample_fusion(agg_out_dim, 256, agg_out_dim, batchnorm)

        self.final_conv = nn.Sequential(nn.Conv2d(agg_out_dim, agg_out_dim // 4, 3, padding=1, bias=False),
                                        batchnorm(agg_out_dim // 4),
                                        nn.ReLU(True),
                                        nn.Conv2d(agg_out_dim // 4, n_class, 1))

        self.resaux_conv = nn.Sequential(
            nn.Conv2d(1024, 1024 // 4, kernel_size=3, stride=1, padding=1, bias=False),
            batchnorm(1024 // 4),
            nn.ReLU(True),
            # nn.Dropout2d(0.1),
            nn.Conv2d(1024 // 4, n_class, 1))
        self.aggaux_conv = nn.Sequential(
            nn.Conv2d(agg_out_dim, agg_out_dim // 4, kernel_size=3, stride=1, padding=1, bias=False),
            batchnorm(agg_out_dim // 4),
            nn.ReLU(True),
            # nn.Dropout2d(0.1),
            nn.Conv2d(agg_out_dim // 4, n_class, 1))

    def forward(self, x):
        imsize = x.size()[2:]
        if not self.is_trian:
            imsize = self.test_size
        c1, c2, c3, c4 = self.base_forward(x)
        contrast_feats1 = self.contrast_extraction1(c1)
        contrast_feats2 = self.contrast_extraction2(c2)
        contrast_feats3 = self.contrast_extraction3(c3)
        contrast_feats4 = self.contrast_extraction4(c4)

        spp_feats = self.spp(c4)

        d4 = self.upsample_fusion4(spp_feats, contrast_feats4)
        d3 = self.upsample_fusion3(d4, contrast_feats3)
        dense_d3 = self.dense_conv3(torch.cat([d3, d4], dim=1))
        d2 = self.upsample_fusion2(dense_d3, contrast_feats2)
        dense_d2 = self.dense_conv2(torch.cat([d2, d3, d4], dim=1))
        d1 = self.upsample_fusion1(dense_d2, contrast_feats1)

        out = self.final_conv(d1)
        out = F.interpolate(out, imsize, mode='bilinear', align_corners=True)

        if self.resaux and not self.aggaux:
            resaux_loss = self.resaux_conv(c3)
            resaux_loss = F.interpolate(resaux_loss, imsize, mode='bilinear', align_corners=True)
            outputs = [out, out, resaux_loss]
        elif not self.resaux and self.aggaux:
            aggaux_loss = self.aggaux_conv(spp_feats)
            aggaux_loss = F.interpolate(aggaux_loss, imsize, mode='bilinear', align_corners=True)
            outputs = [out, out, aggaux_loss]
        elif self.resaux and self.aggaux:
            resaux_loss = self.resaux_conv(c3)
            resaux_loss = F.interpolate(resaux_loss, imsize, mode='bilinear', align_corners=True)
            aggaux_loss = self.aggaux_conv(spp_feats)
            aggaux_loss = F.interpolate(aggaux_loss, imsize, mode='bilinear', align_corners=True)
            outputs = [out, out, resaux_loss, aggaux_loss]
        elif not self.resaux and not self.aggaux:
            outputs = [out, out]

        return tuple(outputs)


class PSPnet(BaseNet):
    def __init__(self, n_class, backbone, aux=False, se_loss=False, resaux=False, aggaux=False, up_conv=0,
                 batchnorm=nn.BatchNorm2d, activation='relu', is_train=True, aspp_rates=[6, 12, 18], boundary_rf=False,
                 boundary_kernel=5, test_size=[256, 256], ensemble=False, aspp_out_dim=512, agg_out_dim=128,
                 spp_size=None, refine=False, refine_ver='v2', **kwargs):
        super(PSPnet, self).__init__(n_class, backbone, aux, se_loss, batchnorm=batchnorm, **kwargs)
        self.is_trian = is_train
        self.test_size = test_size
        self.ensemble = ensemble
        self.resaux = resaux
        self.aggaux = aggaux
        inter_channels = 2048 // 4
        self.spp = PyramidPooling(2048, batchnorm, self._up_kwargs)

        self.final_conv = nn.Sequential(nn.Conv2d(2048 * 2, inter_channels, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(inter_channels),
                                        nn.ReLU(True),
                                        nn.Dropout2d(0.1, False),
                                        nn.Conv2d(inter_channels, n_class, 1))

        self.resaux_conv = nn.Sequential(
            nn.Conv2d(1024, 1024 // 4, kernel_size=3, stride=1, padding=1, bias=False),
            batchnorm(1024 // 4),
            nn.ReLU(True),
            # nn.Dropout2d(0.1),
            nn.Conv2d(1024 // 4, n_class, 1))
        self.aggaux_conv = nn.Sequential(
            nn.Conv2d(2048 * 2, 2048 * 2 // 4, kernel_size=3, stride=1, padding=1, bias=False),
            batchnorm(2048 * 2 // 4),
            nn.ReLU(True),
            # nn.Dropout2d(0.1),
            nn.Conv2d(2048 * 2 // 4, n_class, 1))

    def forward(self, x):
        imsize = x.size()[2:]
        if not self.is_trian:
            imsize = self.test_size
        _, _, c3, c4 = self.base_forward(x)

        spp_feats = self.spp(c4)

        out = self.final_conv(spp_feats)
        out = F.interpolate(out, imsize, mode='bilinear', align_corners=True)

        if self.resaux and not self.aggaux:
            resaux_loss = self.resaux_conv(c3)
            resaux_loss = F.interpolate(resaux_loss, imsize, mode='bilinear', align_corners=True)
            outputs = [out, out, resaux_loss]
        elif not self.resaux and self.aggaux:
            aggaux_loss = self.aggaux_conv(spp_feats)
            aggaux_loss = F.interpolate(aggaux_loss, imsize, mode='bilinear', align_corners=True)
            outputs = [out, out, aggaux_loss]
        elif self.resaux and self.aggaux:
            resaux_loss = self.resaux_conv(c3)
            resaux_loss = F.interpolate(resaux_loss, imsize, mode='bilinear', align_corners=True)
            aggaux_loss = self.aggaux_conv(spp_feats)
            aggaux_loss = F.interpolate(aggaux_loss, imsize, mode='bilinear', align_corners=True)
            outputs = [out, out, resaux_loss, aggaux_loss]
        elif not self.resaux and not self.aggaux:
            outputs = [out, out]

        return tuple(outputs)


def get_exp1_mscm(dataset='pascal_voc', backbone='resnet50', root='./pretrain_models', **kwargs):
    from encoding.dataset import datasets  # , VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = MultiScaleContrastModel(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)

    return model


def get_exp2_densemscm(dataset='pascal_voc', backbone='resnet50', root='./pretrain_models', **kwargs):
    from encoding.dataset import datasets  # , VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = DenseMultiScaleContrastModel(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)

    return model


def get_exp2_mscm_v2(dataset='pascal_voc', backbone='resnet50', root='./pretrain_models', **kwargs):
    from encoding.dataset import datasets  # , VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = MultiScaleContrastModel_v2(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)

    return model


def get_exp3_mscm_v2_wo_edge(dataset='pascal_voc', backbone='resnet50', root='./pretrain_models', **kwargs):
    from encoding.dataset import datasets  # , VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = MultiScaleContrastModel_wo_edge(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)

    return model


def get_pspnet(dataset='pascal_voc', backbone='resnet50', root='./pretrain_models', **kwargs):
    from encoding.dataset import datasets  # , VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = PSPnet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)

    return model
