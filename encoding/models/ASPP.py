# -*- coding: utf-8 -*-
# @Time    : 2019/8/1 18:30
# @Author  : SamChen
# @File    : ASPP.py


import torch
import torch.nn as nn
import torch.nn.functional as F


class _ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, atrous_rate, norm_layer=nn.BatchNorm2d):
        super(_ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=atrous_rate,
                      bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class ASPP(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256, atrous_rates=[3, 6, 9], norm_layer=nn.BatchNorm2d):
        super(ASPP, self).__init__()
        self.b0 = _ASPPConv(in_channels, out_channels, kernel_size=1, padding=0, atrous_rate=1, norm_layer=norm_layer)
        self.b1 = _ASPPConv(in_channels, out_channels, kernel_size=3, padding=atrous_rates[0],
                            atrous_rate=atrous_rates[0],
                            norm_layer=norm_layer)
        self.b2 = _ASPPConv(in_channels, out_channels, kernel_size=3, padding=atrous_rates[1],
                            atrous_rate=atrous_rates[1],
                            norm_layer=norm_layer)
        self.b3 = _ASPPConv(in_channels, out_channels, kernel_size=3, padding=atrous_rates[2],
                            atrous_rate=atrous_rates[2],
                            norm_layer=norm_layer)
        self.b4 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     norm_layer(out_channels),
                                     nn.ReLU(True), )
        # nn.Dropout2d(0.5))

    def forward(self, x):
        feat_size = x.size()[2:]
        feat0 = self.b0(x)
        # print(feat0.size())
        feat1 = self.b1(x)
        # print(feat1.size())
        feat2 = self.b2(x)
        # print(feat2.size())
        feat3 = self.b3(x)
        # print(feat3.size())
        feat4 = self.b4(x)
        # print(feat4.size())
        feat4 = F.interpolate(feat4, feat_size, mode='bilinear', align_corners=True)
        # print(feat4.size())

        x = torch.cat([feat0, feat1, feat2, feat3, feat4], dim=1)
        x = self.project(x)

        return x


class ASPP_v2(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256, atrous_rates=[3, 6, 12, 18], norm_layer=nn.BatchNorm2d):
        super(ASPP_v2, self).__init__()
        self.b0 = _ASPPConv(in_channels, out_channels, kernel_size=1, padding=0, atrous_rate=1, norm_layer=norm_layer)
        self.b1 = _ASPPConv(in_channels, out_channels, kernel_size=3, padding=atrous_rates[0],
                            atrous_rate=atrous_rates[0],
                            norm_layer=norm_layer)
        self.b2 = _ASPPConv(in_channels, out_channels, kernel_size=3, padding=atrous_rates[1],
                            atrous_rate=atrous_rates[1],
                            norm_layer=norm_layer)
        self.b3 = _ASPPConv(in_channels, out_channels, kernel_size=3, padding=atrous_rates[2],
                            atrous_rate=atrous_rates[2],
                            norm_layer=norm_layer)
        self.b4 = _ASPPConv(in_channels, out_channels, kernel_size=3, padding=atrous_rates[3],
                            atrous_rate=atrous_rates[3],
                            norm_layer=norm_layer)
        self.b5 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.project = nn.Sequential(nn.Conv2d(6 * out_channels, out_channels, 1, bias=False),
                                     norm_layer(out_channels),
                                     nn.ReLU(True), )
        # nn.Dropout2d(0.5))

    def forward(self, x):
        feat_size = x.size()[2:]
        feat0 = self.b0(x)
        # print(feat0.size())
        feat1 = self.b1(x)
        # print(feat1.size())
        feat2 = self.b2(x)
        # print(feat2.size())
        feat3 = self.b3(x)
        # print(feat3.size())
        feat4 = self.b4(x)
        # print(feat4.size())
        feat5 = self.b5(x)
        feat5 = F.interpolate(feat5, feat_size, mode='bilinear', align_corners=True)
        # print(feat4.size())

        x = torch.cat([feat0, feat1, feat2, feat3, feat4, feat5], dim=1)
        x = self.project(x)

        return x


class ASPP_depthwise(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256, atrous_rates=[3, 6, 9], norm_layer=nn.BatchNorm2d):
        super(ASPP_depthwise, self).__init__()
        self.b0 = _ASPPConv(in_channels, out_channels, kernel_size=1, padding=0, atrous_rate=1, norm_layer=norm_layer)
        self.b1 = _ASPPConv(in_channels, out_channels, kernel_size=3, padding=atrous_rates[0],
                            atrous_rate=atrous_rates[0],
                            norm_layer=norm_layer)
        self.b2 = _ASPPConv(in_channels, out_channels, kernel_size=3, padding=atrous_rates[1],
                            atrous_rate=atrous_rates[1],
                            norm_layer=norm_layer)
        self.b3 = _ASPPConv(in_channels, out_channels, kernel_size=3, padding=atrous_rates[2],
                            atrous_rate=atrous_rates[2],
                            norm_layer=norm_layer)
        self.b4 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.project = nn.Sequential(nn.ReLU(True),
                                     nn.Conv2d(5 * out_channels, 5 * out_channels, 3, padding=1, bias=False, groups=5 * out_channels),
                                     norm_layer(5 * out_channels),
                                     nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),)
        # nn.Dropout2d(0.5))

    def forward(self, x):
        feat_size = x.size()[2:]
        feat0 = self.b0(x)
        # print(feat0.size())
        feat1 = self.b1(x)
        # print(feat1.size())
        feat2 = self.b2(x)
        # print(feat2.size())
        feat3 = self.b3(x)
        # print(feat3.size())
        feat4 = self.b4(x)
        # print(feat4.size())
        feat4 = F.interpolate(feat4, feat_size, mode='bilinear', align_corners=True)
        # print(feat4.size())

        x = torch.cat([feat0, feat1, feat2, feat3, feat4], dim=1)
        x = self.project(x)

        return x


class ASPP_depthwise_v2(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256, atrous_rates=[3, 6, 9], norm_layer=nn.BatchNorm2d):
        super(ASPP_depthwise_v2, self).__init__()
        self.b0 = _ASPPConv(in_channels, out_channels, kernel_size=1, padding=0, atrous_rate=1, norm_layer=norm_layer)
        self.b1 = _ASPPConv(in_channels, out_channels, kernel_size=3, padding=atrous_rates[0],
                            atrous_rate=atrous_rates[0],
                            norm_layer=norm_layer)
        self.b2 = _ASPPConv(in_channels, out_channels, kernel_size=3, padding=atrous_rates[1],
                            atrous_rate=atrous_rates[1],
                            norm_layer=norm_layer)
        self.b3 = _ASPPConv(in_channels, out_channels, kernel_size=3, padding=atrous_rates[2],
                            atrous_rate=atrous_rates[2],
                            norm_layer=norm_layer)
        self.b4 = _ASPPConv(in_channels, out_channels, kernel_size=3, padding=atrous_rates[3],
                            atrous_rate=atrous_rates[3],
                            norm_layer=norm_layer)
        self.b5 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.project = nn.Sequential(nn.ReLU(True),
                                     nn.Conv2d(6 * out_channels, 6 * out_channels, 3, padding=1, bias=False, groups=6 * out_channels),
                                     norm_layer(6 * out_channels),
                                     nn.Conv2d(6 * out_channels, out_channels, 1, bias=False),)
        # nn.Dropout2d(0.5))

    def forward(self, x):
        feat_size = x.size()[2:]
        feat0 = self.b0(x)
        # print(feat0.size())
        feat1 = self.b1(x)
        # print(feat1.size())
        feat2 = self.b2(x)
        # print(feat2.size())
        feat3 = self.b3(x)
        # print(feat3.size())
        feat4 = self.b4(x)
        # print(feat4.size())
        feat5 = self.b5(x)
        feat4 = F.interpolate(feat5, feat_size, mode='bilinear', align_corners=True)
        # print(feat4.size())

        x = torch.cat([feat0, feat1, feat2, feat3, feat4, feat5], dim=1)
        x = self.project(x)

        return x
