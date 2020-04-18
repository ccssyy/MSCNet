# -*- coding: utf-8 -*-
# @Time    : 2019/12/30 10:16
# @Author  : SamChen
# @File    : __init__.py.py

# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

from .batchnorm import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d
from .replicate import DataParallelWithCallback, patch_replication_callback