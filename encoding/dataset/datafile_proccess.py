# -*- coding: utf-8 -*-
# @Time    : 2019/7/4 10:23
# @Author  : SamChen
# @File    : datafile_proccess.py
import os
import random
import numpy as np
from PIL import Image, ImageOps, ImageChops
import cv2

data_root = '../../Data/ISIC2018_data/'


def split_datafile(root, ratio=0.7):
    ids = [img_file.split('.')[0].split('_')[-1] for img_file in os.listdir(root + 'Image')]
    train_ids = np.random.choice(ids, size=int(len(ids) * ratio), replace=False)
    val_ids = np.array([ids for ids in ids if ids not in train_ids])
    with open(root+'train_file.txt', mode='w+', encoding='utf-8') as train_f:
        for id in np.sort(train_ids):
            train_f.write('Image/'+'ISIC_'+id+'.jpg\t')
            train_f.write('GT/'+'ISIC_'+id+'_segmentation.png\n')
    with open(root + 'val_file.txt', mode='w+', encoding='utf-8') as val_f:
        for id in np.sort(val_ids):
            val_f.write('Image/' + 'ISIC_' + id + '.jpg\t')
            val_f.write('GT/' + 'ISIC_' + id + '_segmentation.png\n')


if __name__ == '__main__':
    # split_datafile(data_root)
    img = Image.open('../../Data/ISIC2017/train/Image/ISIC_0000000.jpg')
    shift_img = ImageChops.offset(img, -20)
    print(img.size[0] * 1.1)
    print(shift_img.size)
    img.show()
    shift_img.show()