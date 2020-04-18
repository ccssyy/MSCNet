# -*- coding: utf-8 -*-
# @Time    : 2019/7/20 22:31
# @Author  : SamChen
# @File    : test.py


from scipy.ndimage import zoom
import scipy.misc as misc
import ast
import os
import sys

cwd = os.getcwd()
root = os.path.split(os.path.split(cwd)[0])[0]
sys.path.append(root)

import numpy as np
from torch.nn.parallel.scatter_gather import gather
from tqdm import tqdm
from sklearn import metrics
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torchvision.transforms as transform
import torch.backends.cudnn as cudnn

from encoding import utils as utils
from encoding.models import get_segmentation_model
from encoding.dataset import get_segmentation_dataset
from encoding.models.customize import SegmentationLosses
from encoding.models.misc import crf_refine
# from encoding.models.syncbn import BatchNorm2d
from encoding.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import argparse
import pandas as pd


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def logger(file='/log.txt', str=None):
    with open(file, mode='a', encoding='utf-8') as f:
        f.write(str + '\n')


class Tester():
    def __init__(self, args):
        self.args = args
        self.log_file = args.resume_dir + '/' + args.log_file
        input_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize([.485, .456, .406], [.229, .224, .225])])
        # dataset
        data_kwargs = {'transform': input_transform, 'img_size': args.img_size}
        # trainset = get_segmentation_dataset(args.dataset, mode='train', augment=True, **data_kwargs)
        # valset = get_segmentation_dataset(args.dataset, mode='val', augment=False, **data_kwargs)
        testset = get_segmentation_dataset(args.testdata, mode='test', augment=False, whole_image=args.whole_image,
                                           **data_kwargs)
        test_vis_set = get_segmentation_dataset(args.testdata, mode='vis', augment=False, whole_image=args.whole_image,
                                                **data_kwargs)
        # dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        # self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size, drop_last=True, shuffle=True, **kwargs)
        # self.valloader = data.DataLoader(valset, batch_size=args.batch_size, drop_last=False, shuffle=False, **kwargs)
        self.testloader = data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                          **kwargs)
        self.visloader = data.DataLoader(test_vis_set, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                         **kwargs)
        # self.nclass = trainset.NUM_CLASS
        # model
        model = get_segmentation_model(args.model, dataset=args.dataset, backbone=args.backbone,
                                       pretrained=args.pretrained, aux=args.aux, is_train=args.is_train,
                                       se_loss=args.se_loss, batchnorm=SynchronizedBatchNorm2d,
                                       img_size=args.img_size, dilated=args.dilated, deep_base=args.deep_base,
                                       multi_grid=args.multi_grid, multi_dilation=args.multi_dilation,
                                       ensemble=args.ensemble, resaux=args.resaux, aggaux=args.aggaux,
                                       output_stride=args.output_stride, test_size=args.test_img_size,
                                       high_rates=args.high_rates, aspp_rates=args.aspp_rates,
                                       aspp_out_dim=args.aspp_out_dim, agg_out_dim=args.agg_out_dim,
                                       up_conv=args.up_conv, refine=args.refine, refine_ver=args.refine_ver)
        # print(model)
        # optimizer using different LR
        # params_list = [{'params': model.pretrain_model.parameters(), 'lr': args.lr},
        #                {'params': model.head.parameters(), 'lr': args.lr * args.head_lr_factor}]

        # optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        # optimizer = torch.optim.Adam(params_list, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

        # clear start epoch if fine-turning
        # if args.ft:
        #     args.start_epoch = 0
        # criterions
        '''
        self.criterion = SegmentationLosses(se_loss=args.se_loss, aux=args.aux, nclass=self.nclass,
                                            xentropy_weight=args.xentropy_weight, mae_weight=args.mae_weight,
                                            dice_weight=args.dice_weight, lovasz_weight=args.lovasz_weight, loss1_4=args.loss1_4,
                                            lov1_4=args.lov1_4, margin_weight=args.margin_weight, margin=args.margin,
                                            k=args.k, c=args.c)
        self.model, self.optimizer = model, optimizer
        '''
        self.model = model
        # using cuda
        # cudnn.benchmark = True
        self.model = torch.nn.DataParallel(self.model).cuda()
        # self.criterion = self.criterion.cuda()
        self.sigmoid = nn.Sigmoid().cuda()
        # lr_scheduler
        # self.scheduler = utils.LR_Scheduler(args.lr_mode, args.lr, args.epochs, len(self.trainloader))
        self.best_pred = 0.0
        # resuming chechpoint
        if args.resume_dir is not None:
            if not os.path.isfile(args.resume_dir + '/checkpoint.pth.tar'):
                print('=> no chechpoint found at {}'.format(args.resume_dir))
                logger(self.log_file, '=> no chechpoint found at {}'.format(args.resume_dir))
            else:
                checkpoint = torch.load(args.resume_dir + '/checkpoint.pth.tar')
                args.start_epoch = checkpoint['epoch']
                self.model.module.load_state_dict(checkpoint['state_dict'])
                if not args.ft:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.best_pred = checkpoint['best_pred']
                print('=> loaded checkpoint {0} (epoch {1})'.format(args.resume_dir, checkpoint['epoch']))
                logger(self.log_file,
                       '=> loaded checkpoint {0} (epoch {1})'.format(args.resume_dir, checkpoint['epoch']))

    def evaluation(self, scales):
        def multiscale_test(model, img, target, origin_img, test_size, scales, test_mode='flip'):
            # image = img.cpu().numpy()
            image = img
            B, C, H, W = image.shape
            pred = torch.zeros([B, 2, test_size[0], test_size[1]]).cuda()
            for scale in scales:
                scale = float(scale)
                # scale_image = zoom(image, (1, 1, scale, scale), order=1, prefilter=False)
                # logist = model(torch.from_numpy(scale_image).cuda())[0]
                scale_image = F.interpolate(image,
                                            (int(self.args.img_size[0] * scale), int(self.args.img_size[1] * scale)),
                                            mode='bilinear', align_corners=True)
                logist = model(scale_image)[0]
                prob = logist.softmax(1)
                if test_mode == 'mirror':
                    flip_logist = model(scale_image.flip(3))[0]
                    flip_prob = flip_logist.softmax(1).flip(3)
                    # flip_prob = flip_prob.flip(3)
                    prob = (prob + flip_prob) * 0.5
                elif test_mode == 'flip':
                    mirror_logist = model(scale_image.flip(2))[0]
                    mirror_prob = mirror_logist.softmax(1).flip(2)
                    # flip_prob = flip_prob.flip(3)
                    prob = (prob + mirror_prob) * 0.5
                elif test_mode == 'mirror+flip':
                    mf_logist = model(scale_image.flip(3).flip(2))[0]
                    mf_prob = mf_logist.softmax(1).flip(2).flip(3)
                    # flip_prob = flip_prob.flip(3)
                    prob = (prob + mf_prob) * 0.5
                elif test_mode is None:
                    logist = model(scale_image)[0]
                    prob = logist.softmax(1)
                pred += prob / len(scales)
                # pred = (prob > pred).float()*prob + (pred >= prob).float()*pred

            batch_size, batch_acc, batch_dice, batch_jacc, batch_sensitivity, batch_specificity = utils.batch_sores(
                pred.detach(), target, origin_img)
            return pred, batch_size, batch_acc, batch_dice, batch_jacc, batch_sensitivity, batch_specificity

        self.model.eval()

        test_num_img = 0
        test_sum_acc = 0
        test_sum_dice = 0
        test_sum_jacc = 0
        test_sum_se = 0
        test_sum_sp = 0

        test_tbar = tqdm(self.testloader, desc='\r')
        for i, (image, target, origin_img) in enumerate(test_tbar):
            with torch.no_grad():
                pred_prob, test_batch_size, test_batch_acc, test_batch_dice, test_batch_jacc, test_batch_sensitivity, test_batch_specificity = multiscale_test(
                    self.model, image, target, origin_img, self.args.test_img_size, scales=scales,
                    test_mode=self.args.test_mode)

            test_num_img += test_batch_size
            test_sum_acc += test_batch_acc
            test_sum_dice += test_batch_dice
            test_sum_jacc += test_batch_jacc
            test_sum_se += test_batch_sensitivity
            test_sum_sp += test_batch_specificity

            test_avg_acc = test_sum_acc / test_num_img
            test_avg_dice = test_sum_dice / test_num_img
            test_avg_jacc = test_sum_jacc / test_num_img
            test_avg_sensitivity = test_sum_se / test_num_img
            test_avg_specificity = test_sum_sp / test_num_img
            test_tbar.set_description(
                'Test      : JA: {0:.4f}, DI: {1:.4f}, SE: {2:.4f}, SP: {3:.4f}, AC: {4:.4f}, img_num: {5}'.format(
                    test_avg_jacc, test_avg_dice, test_avg_sensitivity, test_avg_specificity, test_avg_acc,
                    test_num_img))

        logger(self.log_file,
               'Test      : JA: {0:.4f}, DI: {1:.4f}, SE: {2:.4f}, SP: {3:.4f}, AC: {4:.4f}, img_num: {5}'.format(
                   test_avg_jacc, test_avg_dice, test_avg_sensitivity, test_avg_specificity, test_avg_acc,
                   test_num_img))
        '''
        with open(args.resume_dir + '/c2.txt', mode='a', encoding='utf-8') as f:
            f.write('Test      : JA: {0:.4f}, DI: {1:.4f}, SE: {2:.4f}, SP: {3:.4f}, AC: {4:.4f}, img_num: {5}\n'.format(
                    test_avg_jacc, test_avg_dice, test_avg_sensitivity, test_avg_specificity, test_avg_acc,
                    test_num_img))
        '''
        return test_sum_jacc / test_num_img

    def save_predict_image(self, dataset, scales, ss=False, single_flip=False):
        def multiscale_predict(model, img, test_size, scales, test_mode='flip'):
            # image = img.cpu().numpy()
            image = img
            B, C, H, W = image.shape
            pred = torch.zeros([B, 2, test_size[0], test_size[1]]).cuda()
            for scale in scales:
                scale = float(scale)
                # scale_image = zoom(image, (1, 1, scale, scale), order=1, prefilter=False)
                # logist = model(torch.from_numpy(scale_image).cuda())[0]
                scale_image = F.interpolate(image,
                                            (int(self.args.img_size[0] * scale), int(self.args.img_size[1] * scale)),
                                            mode='bilinear', align_corners=True)
                logist = model(scale_image)[0]
                prob = logist.softmax(1)
                if test_mode == 'mirror':
                    flip_logist = model(scale_image.flip(3))[0]
                    flip_prob = flip_logist.softmax(1).flip(3)
                    # flip_prob = flip_prob.flip(3)
                    prob = (prob + flip_prob) * 0.5
                elif test_mode == 'flip':
                    mirror_logist = model(scale_image.flip(2))[0]
                    mirror_prob = mirror_logist.softmax(1).flip(2)
                    # flip_prob = flip_prob.flip(3)
                    prob = (prob + mirror_prob) * 0.5
                elif test_mode == 'mirror+flip':
                    mf_logist = model(scale_image.flip(3).flip(2))[0]
                    mf_prob = mf_logist.softmax(1).flip(2).flip(3)
                    # flip_prob = flip_prob.flip(3)
                    prob = (prob + mf_prob) * 0.5
                elif test_mode is None:
                    logist = model(scale_image)[0]
                    prob = logist.softmax(1)
                    # flip_prob = flip_prob.flip(3)
                pred += prob / len(scales)
            pred = pred.cpu().numpy()

            return B, pred

        self.model.eval()

        num_img = 0
        image_id_list = []
        checkname = self.args.checkname
        if ss:
            checkname = checkname + '_ss'
        if single_flip:
            checkname = checkname + '_flip'
        with open('../../Data/' + dataset + '/test_file.txt') as f:
            for line in f:
                image_id_list.append(os.path.split(line.split('\t')[0])[-1].split('.')[0])
        if not os.path.exists('../../predict_image/' + dataset + '/' + checkname):
            os.makedirs('../../predict_image/' + dataset + '/' + checkname)
        predict_tbar = tqdm(self.visloader, desc='\r')
        for i, (image, target, origin_img, image_ids) in enumerate(predict_tbar):
            # print(image_id)
            with torch.no_grad():
                if single_flip:
                    predict_batch_size, batch_pred = multiscale_predict(self.model, image, self.args.test_img_size,
                                                                        scales=scales, test_mode='flip')
                else:
                    predict_batch_size, batch_pred = multiscale_predict(self.model, image, self.args.test_img_size,
                                                                        scales=scales, test_mode=self.args.test_mode)

            for j in range(predict_batch_size):
                image_id = os.path.split(image_ids[j])[-1].split('.')[0]
                pred = np.squeeze(batch_pred[j])
                pred = np.argmax(pred, axis=0).astype(np.uint8)
                pred_image = Image.fromarray(pred * 255)
                pred_image.save(
                    '../../predict_image/' + dataset + '/' + checkname + '/' + image_id + '_predict.png')
                target_im = np.squeeze(target[j].cpu().numpy()).astype(np.uint8)
                target_im = Image.fromarray(target_im * 255)
                target_im.save(
                    '../../predict_image/' + dataset + '/' + checkname + '/' + image_id + '_GT.png')

            num_img += predict_batch_size

            predict_tbar.set_description('Saving predict  : img_num: {0}'.format(num_img))

    def save_JA(self, dataset, scales, columns):
        def multiscale_predict(model, img, target, origin_img, test_size, scales, test_mode='flip'):
            # image = img.cpu().numpy()
            image = img
            B, C, H, W = image.shape
            pred = torch.zeros([B, 2, test_size[0], test_size[1]]).cuda()
            for scale in scales:
                scale = float(scale)
                # scale_image = zoom(image, (1, 1, scale, scale), order=1, prefilter=False)
                # logist = model(torch.from_numpy(scale_image).cuda())[0]
                scale_image = F.interpolate(image,
                                            (int(self.args.img_size[0] * scale), int(self.args.img_size[1] * scale)),
                                            mode='bilinear', align_corners=True)
                logist = model(scale_image)[0]
                prob = logist.softmax(1)
                if test_mode == 'mirror':
                    flip_logist = model(scale_image.flip(3))[0]
                    flip_prob = flip_logist.softmax(1).flip(3)
                    # flip_prob = flip_prob.flip(3)
                    prob = (prob + flip_prob) * 0.5
                elif test_mode == 'flip':
                    mirror_logist = model(scale_image.flip(2))[0]
                    mirror_prob = mirror_logist.softmax(1).flip(2)
                    # flip_prob = flip_prob.flip(3)
                    prob = (prob + mirror_prob) * 0.5
                elif test_mode == 'mirror+flip':
                    mf_logist = model(scale_image.flip(3).flip(2))[0]
                    mf_prob = mf_logist.softmax(1).flip(2).flip(3)
                    # flip_prob = flip_prob.flip(3)
                    prob = (prob + mf_prob) * 0.5
                elif test_mode is None:
                    logist = model(scale_image)[0]
                    prob = logist.softmax(1)
                pred += prob / len(scales)
                # pred = (prob > pred).float()*prob + (pred >= prob).float()*pred

            batch_size, batch_jacc, batch_di = utils.batch_ja(pred.detach(), target, origin_img)
            return pred, batch_size, batch_jacc, batch_di

        self.model.eval()

        num_img = 0
        image_id_list = []
        checkname = self.args.checkname
        with open('../../Data/' + dataset + '/test_file.txt') as f:
            for line in f:
                image_id_list.append(os.path.split(line.split('\t')[0])[-1].split('.')[0])
        if not os.path.exists('../../predict_JA/' + dataset + '/' + checkname):
            os.makedirs('../../predict_JA/' + dataset + '/' + checkname)
        all_JA = np.array([])
        all_DI = np.array([])
        predict_tbar = tqdm(self.visloader, desc='\r')
        for i, (image, target, origin_img, image_ids) in enumerate(predict_tbar):
            # print(image_id)
            with torch.no_grad():
                batch_pred, predict_batch_size, batch_ja, batch_di = multiscale_predict(self.model, image, target, origin_img,
                                                                                  self.args.test_img_size,
                                                                                  scales=scales, test_mode=self.args.test_mode)

            all_JA = np.append(all_JA, batch_ja)
            all_DI = np.append(all_DI, batch_di)

            num_img += predict_batch_size

            predict_tbar.set_description('Saving predict  : img_num: {0}'.format(num_img))

        ja_df = pd.DataFrame(all_JA, columns=[columns])
        di_df = pd.DataFrame(all_DI, columns=[columns])
        ja_df.to_csv('../../predict_JA/' + dataset + '/' + checkname + '/all_jaccard_indexes.csv', float_format='%.3f',
                     encoding='utf-8')
        di_df.to_csv('../../predict_JA/' + dataset + '/' + checkname + '/all_dice.csv', float_format='%.3f',
                     encoding='utf-8')
        isic_df = pd.read_excel('./' + dataset + '.xlsx')
        isic_df = pd.concat([isic_df, ja_df], axis=1)
        isic_df.to_csv('./all_models_' + dataset + '.csv', sep='\t', float_format='%.3f', encoding='utf-8')


def parse_args():
    # Traning setting options
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--is-train', type=ast.literal_eval, default=False, help='use for training or not.')
    parser.add_argument('--save', type=ast.literal_eval, default=False)
    parser.add_argument('--save-JA', type=ast.literal_eval, default=False)
    parser.add_argument('--scale', type=float, nargs='+', default=None)
    parser.add_argument('--columns', type=str, default='GCGNet')
    parser.add_argument('--test-mode', type=str, default='mirror+flip', choices=['mirror', 'flip', 'mirror+flip', None])
    parser.add_argument('--dataset', default='ISIC2017')
    parser.add_argument('--testdata', default='PH2')
    parser.add_argument('--model', default='exp21_gsasnet_aspp_depthwise_v5')
    # parser.add_argument('--resume-dir', type=str, default='cityscapes/gsasnet_model/gsasnet101', metavar='PATH',
    #                     help='path to latest checkpoint (default: cityscapes/gsasnet_model/gsasnet101)')
    parser.add_argument('--backbone', default='resnet101')
    parser.add_argument('--checkname', default='exp-21_ph2_aspp_depthwise_v5_conv1_gsasnet101_1e-4_by_1')
    parser.add_argument('--dilated', type=ast.literal_eval, default=True)
    parser.add_argument('--deep-base', type=ast.literal_eval, default=False)
    parser.add_argument('--img-size', type=int, nargs='+', default=[256, 256])
    parser.add_argument('--test-img-size', type=int, nargs='+', default=[256, 256])
    parser.add_argument('--whole-image', type=ast.literal_eval, default=False)
    parser.add_argument('--multi-grid', type=ast.literal_eval, default=True)
    parser.add_argument('--multi-dilation', type=int, nargs='+', default=[4, 4, 4])
    parser.add_argument('--output-stride', type=int, default=8)
    parser.add_argument('--high-rates', type=int, nargs='+', default=[2, 4])
    parser.add_argument('--aspp-rates', type=int, nargs='+', default=[6, 12, 18])
    parser.add_argument('--agg-out-dim', type=int, default=128)
    parser.add_argument('--aspp-out-dim', type=int, default=512)
    parser.add_argument('--up-conv', type=int, default=1, choices=[0, 1, 3])
    parser.add_argument('--ensemble', type=ast.literal_eval, default=False)
    parser.add_argument('--xentropy-weight', type=float, default=1.0)
    parser.add_argument('--mae-weight', type=float, default=0.)
    parser.add_argument('--dice-weight', type=float, default=0.)
    parser.add_argument('--aux-weight', type=float, default=0.5)
    parser.add_argument('--loss1-4', type=float, default=1.0)
    parser.add_argument('--lovasz-weight', type=float, default=1.0)
    parser.add_argument('--lov1-4', type=float, default=0.0)
    parser.add_argument('--margin-weight', type=float, default=1.0)
    parser.add_argument('--margin', type=float, default=0., help='the margin in margin loss.')
    parser.add_argument('--k', type=int, default=0, help='number of the hardest classified points.')
    parser.add_argument('--c', type=int, default=5, help='number of the closest to hard points.')
    # parser.add_argument('--base-size', type=int, default=430)
    # parser.add_argument('--crop-size', type=int, default=320,
    #                     help='input image size of model (should be multiple of 8)')
    # parser.add_argument('--scale', type=bool, default=True)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                        help='which optimizer to use. (default: adam)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--test-batch-size', type=int, default=16)
    parser.add_argument('--head-lr-factor', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default='poly')
    # parser.add_argument('--lr-step', type=int, default=30)
    parser.add_argument('--se-loss', type=ast.literal_eval, default=False)
    parser.add_argument('--aux', type=ast.literal_eval, default=False)
    parser.add_argument('--resaux', type=ast.literal_eval, default=True)
    parser.add_argument('--aggaux', type=ast.literal_eval, default=False)
    parser.add_argument('--aux-fn', type=str, default='dice', choices=['bce', 'dice'])
    parser.add_argument('--momentum', type=float, default='0.9', metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', type=float, default=1e-4, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--pretrained', type=ast.literal_eval, default=False,
                        help='whether to use pretrained base model (default: True)')
    parser.add_argument('--no-val', type=bool, default=False,
                        help='whether not using validation (default: False)')
    parser.add_argument('--ft', type=bool, default=True, help='whether to fine turning (default: True for training)')
    parser.add_argument('--refine', type=ast.literal_eval, default=False)
    parser.add_argument('--refine-ver', type=str, default='v2', choices=['v1', 'v2'])
    parser.add_argument('--log-file', type=str, default='ph2_test_result_mf.txt')
    parser.add_argument('--resume-dir', type=str,
                        default=parser.parse_args().dataset + '/' + parser.parse_args().model + '_model/' + parser.parse_args().checkname,
                        metavar='PATH')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    logger(args.resume_dir + '/' + args.log_file, ' '.join(sys.argv))
    print(args)
    logger(args.resume_dir + '/' + args.log_file, str(args))

    return args


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    tester = Tester(args)
    print('\nStarting Predicting ... ')
    logger(args.resume_dir + '/' + args.log_file, '\nStarting Predicting ... ')
    scales_dict = {'c': [1.0],
                   'b': [0.5, 0.75, 1.0, 1.25, 1.5],
                   'g': [0.75, 0.875, 1.0, 1.125],
                   'h': [0.75, 1.0, 1.125],
                   'e': [0.75, 1.0, 1.25],
                   'n': [0.875, 1.0, 1.125],
                   'o': [0.875, 1.0, 1.25],
                   'p': [0.875, 1.0, 1.125, 1.25],
                   'i': [0.5, 0.75, 1.0, 1.25],
                   'j': [0.5, 0.75, 1.0, 1.125],
                   'k': [0.625, 0.75, 1.0, 1.25],
                   'l': [0.625, 0.75, 1.0, 1.125],
                   'd': [0.75, 1.0, 1.25, 1.5],
                   'a': [0.75, 1.0, 1.25, 1.5, 1.75],
                   'f': [0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
                   # 'q': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
                   # 'r': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.2],
                   # 's': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25],
                   'm': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
                   }
    scales_dict2 = {'e': [0.125],
                    'g': [0.25],
                    'h': [0.375],
                    'd': [0.5],
                    'a': [0.625],
                    'f': [0.75],
                    'b': [0.875],
                    'c': [1.0],
                    'i': [1.125],
                    'j': [1.25],
                    'k': [1.375],
                    'l': [1.5]
                    }
    scales_dict3 = {'d': [0.875, 1.0, 1.125],
                    # 'f': [0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
                    # 'm': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
                    }
    best_scales = [1.0]
    best_ja = 0.0

    if args.save_JA:
        tester.save_JA(args.testdata, args.scale, args.columns)
        torch.cuda.empty_cache()
        exit(1)

    for scales in scales_dict.values():
        # break
        print('\nTest in scales :{}'.format(scales))
        logger(args.resume_dir + '/' + args.log_file, '\nTest in scales :{}'.format(scales))
        ja = tester.evaluation(scales)
        if ja > best_ja:
            best_ja = ja
            best_scales = scales
        torch.cuda.empty_cache()

    if args.save:
        tester.save_predict_image(args.testdata, best_scales)
        torch.cuda.empty_cache()
        # tester.save_predict_image(args.dataset, [1.0], ss=True)
        # torch.cuda.empty_cache()
        # tester.save_predict_image(args.dataset, [1.0], ss=True, single_flip=True)
        # torch.cuda.empty_cache()
