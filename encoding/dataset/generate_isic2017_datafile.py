# -*- coding: utf-8 -*-
# @Time    : 2019/7/5 16:09
# @Author  : SamChen
# @File    : generate_isic2017_datafile.py
import os
import numpy as np


data_root = '../../Data/ISIC2016/'


def get_datafile(root):
    train_ids = ['_'.join(img.split('.')[0].split('_')[:2]) for img in os.listdir(root + 'train/GT')]
    val_ids = ['_'.join(img.split('.')[0].split('_')[:2]) for img in os.listdir(root + 'val/GT')]
    test_ids = ['_'.join(img.split('.')[0].split('_')[:2]) for img in os.listdir(root + 'test/GT')]

    with open(root+'train_file.txt', 'w', encoding='utf-8') as train_f:
        for id in train_ids:
            train_f.write('train/Image/'+id+'.jpg\t')
            train_f.write('train/GT/'+id+'_Segmentation.png\n')

    with open(root+'val_file.txt', 'w', encoding='utf-8') as val_f:
        for id in val_ids:
            val_f.write('val/Image/'+id+'.jpg\t')
            val_f.write('val/GT/'+id+'_Segmentation.png\n')

    with open(root+'test_file.txt', 'w', encoding='utf-8') as test_f:
        for id in test_ids:
            test_f.write('test/Image/'+id+'.jpg\t')
            test_f.write('test/GT/'+id+'_Segmentation.png\n')


if __name__ == '__main__':
    get_datafile(data_root)