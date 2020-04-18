##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import torch
from sklearn.metrics.pairwise import paired_euclidean_distances, euclidean_distances
from encoding.models.misc import crf_refine
from PIL import Image
import cv2


def batch_pix_accuracy(predict, target):
    """Batch Pixel Accuracy
    Args:
        predict: input 4D tensor
        target: label 3D tensor
    """
    _, predict = torch.max(predict, 1)
    predict = predict.cpu().numpy().astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1
    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target) * (target > 0))
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(predict, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """
    _, predict = torch.max(predict, 1)
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = predict.cpu().numpy().astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1

    predict = predict * (target > 0).astype(predict.dtype)
    intersection = predict * (predict == target)
    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union


def batch_sores(predict, target, image):
    batch_size = predict.size()[0]
    # predict[predict > 0.5] = 1
    # predict[predict != 1] = 0
    _, predict = predict.max(dim=1)
    predict = predict.cpu().numpy().astype('int64')
    target = target.cpu().numpy().astype('int64')
    image = image.cpu().numpy().astype('uint8')
    if image.shape[1] == 3:
        image = np.reshape(image, [image.shape[0], image.shape[2], image.shape[3], image.shape[1]])
    batch_acc = 0
    batch_dice = 0
    batch_jacc = 0
    batch_sensitivity = 0
    batch_specificity = 0
    for i in range(batch_size):
        refine_pred = crf_refine(image[i], predict[i].squeeze().astype('uint8')).astype('int64')
        # refine_pred = predict[i]
        # Image.fromarray(refine_pred * 255.0).show(str(i))
        # Image.fromarray(predict[i] * 255.0).show(str(i))
        # cv2.waitKey(5)
        TP = np.sum(np.logical_and(refine_pred == 1, target[i] == 1))
        TN = np.sum(np.logical_and(refine_pred == 0, target[i] == 0))
        FP = np.sum(np.logical_and(refine_pred == 1, target[i] == 0))
        FN = np.sum(np.logical_and(refine_pred == 0, target[i] == 1))

        # Compute Accuracy score
        pixACC = (TP + TN) / float(TP + TN + FP + FN)
        batch_acc += pixACC

        # Compute Dice coefficient
        intersection = np.logical_and(target[i], refine_pred)
        dice = 2. * intersection.sum() / (refine_pred.sum() + target[i].sum())
        batch_dice += dice

        # Compute Jaccard similarity coefficient score
        jacc = TP / float(TP + FN + FP)
        batch_jacc += jacc

        # Compute the Sensitivity (recall)
        sensitivity = TP / float(TP + FN)
        batch_sensitivity += sensitivity

        # Compute Specificity score
        specificity = TN / float(TN + FP)
        batch_specificity += specificity

    return batch_size, batch_acc, batch_dice, batch_jacc, batch_sensitivity, batch_specificity


def batch_ja(predict, target, image):
    batch_size = predict.size()[0]
    # predict[predict > 0.5] = 1
    # predict[predict != 1] = 0
    _, predict = predict.max(dim=1)
    predict = predict.cpu().numpy().astype('int64')
    target = target.cpu().numpy().astype('int64')
    image = image.cpu().numpy().astype('uint8')
    if image.shape[1] == 3:
        image = np.reshape(image, [image.shape[0], image.shape[2], image.shape[3], image.shape[1]])
    batch_jacc = []
    batch_di = []
    for i in range(batch_size):
        refine_pred = crf_refine(image[i], predict[i].squeeze().astype('uint8')).astype('int64')
        # refine_pred = predict[i]
        # Image.fromarray(refine_pred * 255.0).show(str(i))
        # Image.fromarray(predict[i] * 255.0).show(str(i))
        # cv2.waitKey(5)
        TP = np.sum(np.logical_and(refine_pred == 1, target[i] == 1))
        TN = np.sum(np.logical_and(refine_pred == 0, target[i] == 0))
        FP = np.sum(np.logical_and(refine_pred == 1, target[i] == 0))
        FN = np.sum(np.logical_and(refine_pred == 0, target[i] == 1))

        # Compute Jaccard similarity coefficient score
        jacc = TP / float(TP + FN + FP)
        batch_jacc.append(jacc)

        intersection = np.logical_and(target[i], refine_pred)
        dice = 2. * intersection.sum() / (refine_pred.sum() + target[i].sum())
        batch_di.append(dice)

    return batch_size, np.array(batch_jacc), np.array(batch_di)


# ref https://github.com/CSAILVision/sceneparsing/blob/master/evaluationCode/utils_eval.py
def pixel_accuracy(im_pred, im_lab):
    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)

    # Remove classes from unlabeled pixels in gt image. 
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(im_lab > 0)
    pixel_correct = np.sum((im_pred == im_lab) * (im_lab > 0))
    # pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
    return pixel_correct, pixel_labeled


def intersection_and_union(im_pred, im_lab, num_class):
    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)
    # Remove classes from unlabeled pixels in gt image. 
    im_pred = im_pred * (im_lab > 0)
    # Compute area intersection:
    intersection = im_pred * (im_pred == im_lab)
    area_inter, _ = np.histogram(intersection, bins=num_class - 1,
                                 range=(1, num_class - 1))
    # Compute area union: 
    area_pred, _ = np.histogram(im_pred, bins=num_class - 1,
                                range=(1, num_class - 1))
    area_lab, _ = np.histogram(im_lab, bins=num_class - 1,
                               range=(1, num_class - 1))
    area_union = area_pred + area_lab - area_inter
    return area_inter, area_union
