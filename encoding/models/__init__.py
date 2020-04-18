# -*- coding: utf-8 -*-
# @Time    : 2019/6/21 16:16
# @Author  : SamChen
# @File    : __init__.py.py

from .base import *
from .MultiScaleContrastModel import *


def get_segmentation_model(name, **kwargs):
    models = {
        'exp1_mscm': get_exp1_mscm,
        'exp2_densemscm': get_exp2_densemscm,
        'exp2_mscm_v2': get_exp2_mscm_v2,
        'exp3_mscm_v2_wo_edge': get_exp3_mscm_v2_wo_edge,
        'pspnet': get_pspnet,
    }
    return models[name.lower()](**kwargs)
