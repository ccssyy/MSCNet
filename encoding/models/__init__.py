# -*- coding: utf-8 -*-
# @Time    : 2019/6/21 16:16
# @Author  : SamChen
# @File    : __init__.py.py
from .GSASNet import *
from .base import *
from .MultiScaleContrastModel import *


def get_segmentation_model(name, **kwargs):
    models = {
        'gsasnet': get_gsasnet,
        'exp15_gsasnet': get_exp15_gsasnet,
        'exp16_gsasnet': get_exp16_gsasnet,
        'exp17_gsasnet': get_exp17_gsasnet,
        'exp18_gsasnet': get_exp18_gsasnet,  # use resaux loss or aggaux loss
        'exp18_no_bn_gsasnet': get_exp18_no_BN_gsasnet,
        'exp18_notcatres4_gsasnet': get_exp18_notcatres4_gsasnet,
        'exp19_gsasnet': get_exp19_gsasnet,  # same chennels in both local feature and globel feature
        'exp20_gsasnet': get_exp20_gsasnet,  # use aggregation global_x as the input of GSAHead
        'exp21_gsasnet': get_exp21_gsasnet,  # use global_x without avgpooling and resblock4 to product affinity
        'exp18_aspp_gsasnet': get_exp18_gsasnet_aspp,
        'exp21_aspp_gsasnet': get_exp21_gsasnet_aspp,
        'exp21_aspp_spp_gsasnet': get_exp21_gsasnet_aspp_spp,
        'deeplabv3': get_deeplabv3,
        'exp21_gsasnet_aspp_wo_global': get_exp21_gsasnet_aspp_wo_global,
        'exp21_gsasnet_aspp_wo_affinity': get_exp21_gsasnet_aspp_wo_Affinity,
        'exp21_gsasnet_aspp_depthwise': get_exp21_gsasnet_aspp_depthwise,
        'exp21_gsasnet_aspp_depthwise_v2': get_exp21_gsasnet_aspp_depthwise_v2,
        'exp21_gsasnet_aspp_depthwise_v3': get_exp21_gsasnet_aspp_depthwise_v3,
        'exp21_gsasnet_aspp_depthwise_v4': get_exp21_gsasnet_aspp_depthwise_v4,
        'exp21_gsasnet_aspp_depthwise_v5': get_exp21_gsasnet_aspp_depthwise_v5,
        'exp21_gsasnet_aspp_depthwise_v6': get_exp21_gsasnet_aspp_depthwise_v6,
        'exp21_gsasnet_aspp_depthwise_v7': get_exp21_gsasnet_aspp_depthwise_v7,
        'exp21_gsasnet_aspp_depthwise_v8': get_exp21_gsasnet_aspp_depthwise_v8,
        'exp21_gsasnet_aspp_depthwise_v5_no_affinity': get_exp21_gsasnet_aspp_depthwise_v5_wo_affinity,
        'exp21_gsasnet_aspp_depthwise_v5_no_global': get_exp21_gsasnet_aspp_depthwise_v5_wo_global,
        'exp1_mscm': get_exp1_mscm,
        'exp2_densemscm': get_exp2_densemscm,
        'exp2_mscm_v2': get_exp2_mscm_v2,
        'exp3_mscm_v2_wo_edge': get_exp3_mscm_v2_wo_edge,
        'pspnet': get_pspnet,
    }
    return models[name.lower()](**kwargs)
