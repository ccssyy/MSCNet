# -*- coding: utf-8 -*-
# @Time    : 2019/8/17 14:37
# @Author  : SamChen
# @File    : generate_PH2_datafile.py

import os
import numpy as np

data_root = '../../Data/PH2/'


def generate_file(root):
    image_ids = os.listdir(root)
    with open(root+'test_file.txt', 'w', encoding='utf-8') as f:
        for id in image_ids:
            f.write(root + id + '/' + id + '_Dermoscopic_Image/' + id + '.bmp\t')
            f.write(root + id + '/' + id + '_lesion/' + id + '_lesion.bmp\n')


if __name__ == '__main__':
    generate_file(data_root)