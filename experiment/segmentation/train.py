# -*- coding: utf-8 -*-
# @Time    : 2019/6/21 17:38
# @Author  : SamChen
# @File    : train.py
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

import torch
import torch.nn as nn
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
import shutil


def logger(file='/log.txt', str=None):
    if not os.path.exists(os.path.split(file)[0]):
        os.makedirs(os.path.split(file)[0])
    with open(file, mode='a', encoding='utf-8') as f:
        f.write(str + '\n')


class Trainer():
    def __init__(self, args):
        self.args = args
        self.log_file = args.resume_dir + '/' + args.log_file
        input_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize([.485, .456, .406], [.229, .224, .225])])
        # dataset
        data_kwargs = {'transform': input_transform, 'img_size': args.img_size}
        trainset = get_segmentation_dataset(args.dataset, mode='train', augment=True, data_num=args.data_num,
                                            **data_kwargs)
        valset = get_segmentation_dataset(args.dataset, mode='val', augment=False, **data_kwargs)
        testset = get_segmentation_dataset(args.dataset, mode='test', augment=False, **data_kwargs)
        # dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size, drop_last=False, shuffle=True,
                                           **kwargs)
        self.valloader = data.DataLoader(valset, batch_size=args.batch_size, drop_last=False, shuffle=False, **kwargs)
        self.testloader = data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                          **kwargs)
        self.nclass = trainset.NUM_CLASS
        # model
        model = get_segmentation_model(args.model, dataset=args.dataset, backbone=args.backbone,
                                       pretrained=args.pretrained, aux=args.aux,
                                       se_loss=args.se_loss, batchnorm=SynchronizedBatchNorm2d,
                                       img_size=args.img_size, dilated=args.dilated, deep_base=args.deep_base,
                                       multi_grid=args.multi_grid, multi_dilation=args.multi_dilation,
                                       ensemble=args.ensemble, resaux=args.resaux, aggaux=args.aggaux,
                                       output_stride=args.output_stride, high_rates=args.high_rates,
                                       aspp_rates=args.aspp_rates, aspp_out_dim=args.aspp_out_dim,
                                       agg_out_dim=args.agg_out_dim, up_conv=args.up_conv, boundary_rf=args.boundary_rf,
                                       pretrained_file=args.pretrained_file, refine=args.refine,
                                       boundary_kernel=args.boundary_kernel, spp_size=args.spp_size,
                                       refine_ver=args.refine_ver, activation=args.activation)

        # optimizer using different LR
        if args.model.__contains__('gsas'):
            params_list = [{'params': model.pretrain_model.parameters(), 'lr': args.lr},
                           {'params': model.aggregation.parameters(), 'lr': args.lr},
                           ]
            if args.loss1_4 != 0:
                params_list.append({'params': model.loss1_conv.parameters(), 'lr': args.lr})
                params_list.append({'params': model.loss2_conv.parameters(), 'lr': args.lr})
                params_list.append({'params': model.loss3_conv.parameters(), 'lr': args.lr})
                params_list.append({'params': model.loss4_conv.parameters(), 'lr': args.lr})
            if args.aux:
                params_list.append({'params': model.resaux_conv.parameters(), 'lr': args.lr})
                params_list.append({'params': model.aggaux_conv.parameters(), 'lr': args.lr})
            if args.model.__contains__('aspp'):
                params_list.append({'params': model.aspp.parameters(), 'lr': args.lr})
            if args.refine:
                params_list.extend([{'params': model.rm_global_conv.parameters(), 'lr': args.lr * args.head_lr_factor},
                                    {'params': model.RM.parameters(), 'lr': args.lr * args.head_lr_factor}])

            params_list.append({'params': model.head.parameters(), 'lr': args.lr * args.head_lr_factor})
        elif args.model.__contains__('mscm'):
            params_list = [{'params': model.parameters(), 'lr': args.lr}]
        elif args.model.__contains__('pspnet'):
            params_list = [{'params': model.parameters(), 'lr': args.lr}]

        if args.freezn:
            # params_list = []
            for i, m in enumerate(model.children()):
                if i not in [4, 5]:
                    # print(m)
                    for params in m.parameters():
                        params.requires_grad = False
                # else:
                #     params_list.append({'params': m.parameters(), 'lr': args.lr})
        # print([{'params': [list(filter(lambda p: p.requires_grad, params_list[_]['params']))], 'lr': params_list[_]['lr']} for _ in range(8)])
        # print(list(filter(lambda p: p.requires_grad, model.parameters())))

        if args.optimizer == 'sgd':
            if args.freezn:
                optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                            momentum=args.momentum, weight_decay=args.weight_decay)
            else:
                optimizer = torch.optim.SGD(params_list, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer == 'adam':
            if args.freezn:
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                             betas=(0.9, 0.999))
            else:
                optimizer = torch.optim.Adam(params_list, betas=(0.9, 0.999))
            print(len(optimizer.param_groups))
        # exit(111)
        # criterions
        if args.weight is not None:
            weight = torch.Tensor(args.weight)
        else:
            weight = None
        self.criterion = SegmentationLosses(se_loss=args.se_loss, aux=args.aux, nclass=self.nclass, weight=weight,
                                            xentropy_weight=args.xentropy_weight, mae_weight=args.mae_weight,
                                            dice_weight=args.dice_weight, lovasz_weight=args.lovasz_weight,
                                            loss1_4=args.loss1_4, lov1_4=args.lov1_4, margin_weight=args.margin_weight,
                                            margin=args.margin, k=args.k, c=args.c, ensemble=args.ensemble,
                                            resaux=args.resaux, aggaux=args.aggaux, aux_weight=args.aux_weight,
                                            aux_fn=args.aux_fn, boundary_weight=args.boundary_weight,
                                            boundary_rf=args.boundary_rf, boundary_kernel=args.boundary_kernel,
                                            refine=args.refine, refine_ce_weight=args.refine_ce_weight,
                                            refine_dice_weight=args.refine_dice_weight)
        self.model, self.optimizer = model, optimizer
        # using cuda
        # cudnn.benchmark = True
        self.model = torch.nn.DataParallel(self.model).cuda()
        self.criterion = self.criterion.cuda()
        self.sigmoid = nn.Sigmoid().cuda()
        # lr_scheduler
        if args.lr_scheduler == 'v1':
            self.scheduler = utils.LR_Scheduler(args.lr_mode, args.lr, args.epochs, len(self.trainloader),
                                                freezn=args.freezn, decode_lr_factor=args.head_lr_factor)
        elif args.lr_scheduler == 'v2':  # correct version
            self.scheduler = utils.LR_Scheduler_v2(args.lr_mode, args.lr, args.epochs, len(self.trainloader),
                                                   freezn=args.freezn, decode_lr_factor=args.head_lr_factor)
        self.best_pred = 0.0
        # resuming chechpoint
        if args.resume_dir is not None:
            if not os.path.isfile(args.resume_dir + '/checkpoint.pth.tar'):
                print('=> no chechpoint found at {}'.format(args.resume_dir))
                logger(self.log_file, '=> no chechpoint found at {}'.format(args.resume_dir))
                args.start_epoch = 0
            else:
                if args.freezn:
                    shutil.copyfile(args.resume_dir + '/checkpoint.pth.tar',
                                    args.resume_dir + '/checkpoint_origin.pth.tar')
                    shutil.copyfile(args.resume_dir + '/model_best.pth.tar',
                                    args.resume_dir + '/model_best_origin.pth.tar')
                if not args.ft:
                    checkpoint = torch.load(args.resume_dir + '/checkpoint.pth.tar')
                else:
                    if not os.path.isfile(args.resume_dir + '/fineturned_checkpoint.pth.tar') and not os.path.isfile(
                            args.resume_dir + '/checkpoint.pth.tar'):
                        print('=> no chechpoint found at {}'.format(args.resume_dir))
                        logger(self.log_file, '=> no chechpoint found at {}'.format(args.resume_dir))
                    elif os.path.isfile(args.resume_dir + '/fineturned_checkpoint.pth.tar'):
                        checkpoint = torch.load(args.resume_dir + '/fineturned_checkpoint.pth.tar')
                    else:
                        checkpoint = torch.load(args.resume_dir + '/checkpoint.pth.tar')
                args.start_epoch = checkpoint['epoch']
                self.model.module.load_state_dict(checkpoint['state_dict'], strict=False)
                if not args.ft and not args.freezn:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.best_pred = checkpoint['best_pred']
                print('=> loaded checkpoint {0} (epoch {1})'.format(args.resume_dir, checkpoint['epoch']))
                logger(self.log_file,
                       '=> loaded checkpoint {0} (epoch {1})'.format(args.resume_dir, checkpoint['epoch']))

                # clear start epoch if fine-turning
                if args.ft:
                    args.start_epoch = 0

    def training(self, epoch):
        if self.args.dynamic_lweight:
            if epoch == 0:
                self.criterion.xentropy_weight = 1.0
                self.criterion.lovasz_weight = 0.0
            elif epoch == 20:
                self.criterion.xentropy_weight = 0.75
                self.criterion.lovasz_weight = 0.25
            elif epoch == 45:
                self.criterion.xentropy_weight = 0.5
                self.criterion.lovasz_weight = 0.5
            elif epoch == 70:
                self.criterion.xentropy_weight = 0.25
                self.criterion.lovasz_weight = 0.75
            print('xentropy_weight: {}  lovasz_weight :{}'.format(self.criterion.xentropy_weight,
                                                                  self.criterion.lovasz_weight))
            logger(self.log_file, 'xentropy_weight: {}  lovasz_weight :{}'.format(self.criterion.xentropy_weight,
                                                                                  self.criterion.lovasz_weight))
        train_loss = 0.0
        t_xentropy_loss = 0.0
        if self.args.ensemble:
            t_loss0, t_lov0 = 0.0, 0.0
        t_loss1 = 0.0
        t_loss2 = 0.0
        t_loss3 = 0.0
        t_loss4 = 0.0
        t_lov1 = 0.0
        t_lov2 = 0.0
        t_lov3 = 0.0
        t_lov4 = 0.0
        t_mae_loss = 0.0
        t_dice_loss = 0.0
        t_lovasz_loss = 0.0
        total_mloss = 0.0
        total_bloss = 0.0
        if self.args.refine:
            t_rbce = 0.0
            t_rdice = 0.0
        self.model.train()
        tbar = tqdm(self.trainloader, desc='\r')
        for i, (image, target, _) in enumerate(tbar):
            image = image.cuda()
            target = target.cuda()
            image = torch.autograd.Variable(image)
            target = torch.autograd.Variable(target)
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            logists = self.model(image)
            if self.args.ensemble:
                if self.args.refine:
                    loss, xentropy_loss, bloss, loss1, lov1, loss2, lov2, loss3, lov3, loss4, lov4, loss0, lov0, mae_loss, dice_loss, lovasz_loss, mloss, refine_xentropy, refine_dice = self.criterion(
                        logists, target)
                else:
                    loss, xentropy_loss, bloss, loss1, lov1, loss2, lov2, loss3, lov3, loss4, lov4, loss0, lov0, mae_loss, dice_loss, lovasz_loss, mloss = self.criterion(
                        logists, target)
            else:
                if self.args.refine:
                    loss, xentropy_loss, bloss, loss1, lov1, loss2, lov2, loss3, lov3, loss4, lov4, mae_loss, dice_loss, lovasz_loss, mloss, refine_xentropy, refine_dice = self.criterion(
                        logists, target)
                else:
                    loss, xentropy_loss, bloss, loss1, lov1, loss2, lov2, loss3, lov3, loss4, lov4, mae_loss, dice_loss, lovasz_loss, mloss = self.criterion(
                        logists, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            t_xentropy_loss += xentropy_loss.item()
            if self.args.ensemble:
                t_loss0 += loss0.item()
                t_lov0 += lov0.item()
            t_loss1 += loss1.item()
            t_loss2 += loss2.item()
            t_loss3 += loss3.item()
            t_loss4 += loss4.item()
            t_lov1 += lov1.item()
            t_lov2 += lov2.item()
            t_lov3 += lov3.item()
            t_lov4 += lov4.item()
            t_mae_loss += mae_loss.item()
            t_dice_loss += dice_loss.item()
            t_lovasz_loss += lovasz_loss.item()
            total_mloss += mloss.item()
            total_bloss += bloss.item()
            if self.args.refine:
                t_rbce += refine_xentropy.item()
                t_rdice += refine_dice.item()
            # t_xentropy_loss += 0
            # t_mae_loss += 0
            # t_dice_loss += 0
            # t_lovasz_loss += 0
            # total_mloss += 0
            if self.args.ensemble:
                if self.args.refine:
                    tbar.set_description(
                        'Epoch-{}: Train loss: {:.3f}, xentropy_loss: {:.3f}, bloss: {:.8f}, loss1: {:.3f}, lov1: {:.3f}, loss2: {:.3f}, '
                        'lov2: {:.3f}, mae_loss: {:.3f}, dice_loss: {:.3f}, lovasz_loss: {:.3f}, mloss: {:.4f}, r_ce: {:.4f}, r_dice: {:.3f}'.format(
                            epoch, train_loss / (i + 1), t_xentropy_loss / (i + 1), total_bloss / (i + 1),
                                   t_loss1 / (i + 1), t_lov1 / (i + 1), t_loss2 / (i + 1), t_lov2 / (i + 1),
                                   t_mae_loss / (i + 1), t_dice_loss / (i + 1), t_lovasz_loss / (i + 1),
                                   total_mloss / (i + 1)), t_rbce / (i + 1), t_rdice / (i + 1))
                else:
                    tbar.set_description(
                        'Epoch-{}: Train loss: {:.3f}, xentropy_loss: {:.3f}, bloss: {:.8f}, loss1: {:.3f}, lov1: {:.3f}, loss2: {:.3f}, '
                        'lov2: {:.3f}, mae_loss: {:.3f}, dice_loss: {:.3f}, lovasz_loss: {:.3f}, mloss: {:.4f} '.format(
                            epoch, train_loss / (i + 1), t_xentropy_loss / (i + 1), total_bloss / (i + 1),
                                   t_loss1 / (i + 1), t_lov1 / (i + 1), t_loss2 / (i + 1), t_lov2 / (i + 1),
                                   t_mae_loss / (i + 1), t_dice_loss / (i + 1), t_lovasz_loss / (i + 1),
                                   total_mloss / (i + 1)))
            else:
                if self.args.refine:
                    tbar.set_description(
                        'Epoch-{}: Train loss: {:.3f}, xentropy_loss: {:.3f}, bloss: {:.8f}, loss1: {:.3f}, lov1: {:.3f}, loss2: {:.3f}, '
                        'lov2: {:.3f}, mae_loss: {:.3f}, dice_loss: {:.3f}, lovasz_loss: {:.3f}, mloss: {:.4f}, r_ce: {:.4f}, r_dice: {:.3f}'.format(
                            epoch, train_loss / (i + 1), t_xentropy_loss / (i + 1), total_bloss / (i + 1),
                                   t_loss1 / (i + 1), t_lov1 / (i + 1), t_loss2 / (i + 1), t_lov2 / (i + 1),
                                   t_mae_loss / (i + 1), t_dice_loss / (i + 1), t_lovasz_loss / (i + 1),
                                   total_mloss / (i + 1), t_rbce / (i + 1), t_rdice / (i + 1)))
                else:
                    tbar.set_description(
                        'Epoch-{}: Train loss: {:.3f}, xentropy_loss: {:.3f}, bloss: {:.8f}, loss1: {:.3f}, lov1: {:.3f}, loss2: {:.3f}, '
                        'lov2: {:.3f}, mae_loss: {:.3f}, dice_loss: {:.3f}, lovasz_loss: {:.3f}, mloss: {:.4f} '.format(
                            epoch, train_loss / (i + 1), t_xentropy_loss / (i + 1), total_bloss / (i + 1),
                                   t_loss1 / (i + 1), t_lov1 / (i + 1), t_loss2 / (i + 1), t_lov2 / (i + 1),
                                   t_mae_loss / (i + 1), t_dice_loss / (i + 1),
                                   t_lovasz_loss / (i + 1), total_mloss / (i + 1)))
            # break
        if self.args.ensemble:
            if self.args.refine:
                logger(self.log_file,
                       'Epoch-{}: Train loss: {:.3f}, xentropy_loss: {:.3f}, bloss: {:.8f}, loss1: {:.3f}, lov1: {:.3f}, loss2: {:.3f}, '
                       'lov2: {:.3f}, mae_loss: {:.3f}, dice_loss: {:.3f}, lovasz_loss: {:.3f}, mloss: {:.4f}, r_ce: {:.4f}, r_dice: {:.3f}'.format(
                           epoch, train_loss / (tbar.__len__() + 1), t_xentropy_loss / (tbar.__len__() + 1),
                                  total_bloss / (tbar.__len__() + 1),
                                  t_loss1 / (tbar.__len__() + 1), t_lov1 / (tbar.__len__() + 1),
                                  t_loss2 / (tbar.__len__() + 1), t_lov2 / (tbar.__len__() + 1),
                                  t_mae_loss / (tbar.__len__() + 1), t_dice_loss / (tbar.__len__() + 1),
                                  t_lovasz_loss / (tbar.__len__() + 1), total_mloss / (tbar.__len__() + 1),
                                  t_rbce / (tbar.__len__() + 1), t_rdice / (tbar.__len__() + 1)))
            else:
                logger(self.log_file,
                       'Epoch-{}: Train loss: {:.3f}, xentropy_loss: {:.3f}, bloss: {:.8f}, loss1: {:.3f}, lov1: {:.3f}, loss2: {:.3f}, '
                       'lov2: {:.3f}, mae_loss: {:.3f}, dice_loss: {:.3f}, lovasz_loss: {:.3f}, mloss: {:.4f} '.format(
                           epoch, train_loss / (tbar.__len__() + 1), t_xentropy_loss / (tbar.__len__() + 1),
                                  total_bloss / (tbar.__len__() + 1),
                                  t_loss1 / (tbar.__len__() + 1), t_lov1 / (tbar.__len__() + 1),
                                  t_loss2 / (tbar.__len__() + 1), t_lov2 / (tbar.__len__() + 1),
                                  t_mae_loss / (tbar.__len__() + 1), t_dice_loss / (tbar.__len__() + 1),
                                  t_lovasz_loss / (tbar.__len__() + 1), total_mloss / (tbar.__len__() + 1)))
        else:
            if self.args.refine:
                logger(self.log_file,
                       'Epoch-{}: Train loss: {:.3f}, xentropy_loss: {:.3f}, bloss: {:.8f}, loss1: {:.3f}, lov1: {:.3f}, loss2: {:.3f}, '
                       'lov2: {:.3f}, mae_loss: {:.3f}, dice_loss: {:.3f}, lovasz_loss: {:.3f}, mloss: {:.4f}, r_ce: {:.4f}, r_dice: {:.3f}'.format(
                           epoch, train_loss / (tbar.__len__() + 1), t_xentropy_loss / (tbar.__len__() + 1),
                                  total_bloss / (tbar.__len__() + 1),
                                  t_loss1 / (tbar.__len__() + 1), t_lov1 / (tbar.__len__() + 1),
                                  t_loss2 / (tbar.__len__() + 1), t_lov2 / (tbar.__len__() + 1),
                                  t_mae_loss / (tbar.__len__() + 1), t_dice_loss / (tbar.__len__() + 1),
                                  t_lovasz_loss / (tbar.__len__() + 1), total_mloss / (tbar.__len__() + 1),
                                  t_rbce / (tbar.__len__() + 1), t_rdice / (tbar.__len__() + 1)))
            else:
                logger(self.log_file,
                       'Epoch-{}: Train loss: {:.3f}, xentropy_loss: {:.3f}, bloss: {:.8f}, loss1: {:.3f}, lov1: {:.3f}, loss2: {:.3f}, '
                       'lov2: {:.3f}, mae_loss: {:.3f}, dice_loss: {:.3f}, lovasz_loss: {:.3f}, mloss: {:.4f} '.format(
                           epoch, train_loss / (tbar.__len__() + 1), t_xentropy_loss / (tbar.__len__() + 1),
                                  total_bloss / (tbar.__len__() + 1),
                                  t_loss1 / (tbar.__len__() + 1), t_lov1 / (tbar.__len__() + 1),
                                  t_loss2 / (tbar.__len__() + 1), t_lov2 / (tbar.__len__() + 1),
                                  t_mae_loss / (tbar.__len__() + 1), t_dice_loss / (tbar.__len__() + 1),
                                  t_lovasz_loss / (tbar.__len__() + 1), total_mloss / (tbar.__len__() + 1)))

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred
            }, self.args, is_best)

    def validation_and_test(self, epoch):
        # Fast test during the training
        def eval_batch(model, image, target, origin_img):
            target = target.cuda()
            # image = torch.autograd.Variable(image)
            # target = torch.autograd.Variable(target)
            logists = model(image)
            # outputs = gather(outputs, 0, dim=0)
            if self.args.refine:
                if self.args.boundary_rf:
                    pred = logists[0].softmax(1)
                else:
                    pred = logists[-1].softmax(1)
            else:
                pred = logists[0].softmax(1)
            # correct, labeled = utils.batch_pix_accuracy(pred.data, target)
            # inter, union = utils.batch_intersection_union(pred.data, target, self.nclass)
            batch_size, batch_acc, batch_dice, batch_jacc, batch_sensitivity, batch_specificity = utils.batch_sores(
                pred.detach(), target, origin_img)
            return batch_size, batch_acc, batch_dice, batch_jacc, batch_sensitivity, batch_specificity

        is_best = False
        self.model.eval()

        val_num_img, test_num_img = 0, 0
        val_sum_acc, test_sum_acc = 0, 0
        val_sum_dice, test_sum_dice = 0, 0
        val_sum_jacc, test_sum_jacc = 0, 0
        val_sum_sensitivity, test_sum_se = 0, 0
        val_sum_specificity, test_sum_sp = 0, 0

        test_tbar = tqdm(self.testloader, desc='\r')
        for i, (image, target, origin_img) in enumerate(test_tbar):
            with torch.no_grad():
                test_batch_size, test_batch_acc, test_batch_dice, test_batch_jacc, test_batch_sensitivity, test_batch_specificity = eval_batch(
                    self.model, image, target, origin_img)

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

        self.model.eval()
        val_tbar = tqdm(self.valloader, desc='\r')
        for i, (image, target, origin_img) in enumerate(val_tbar):
            with torch.no_grad():
                val_batch_size, val_batch_acc, val_batch_dice, val_batch_jacc, val_batch_sensitivity, val_batch_specificity = eval_batch(
                    self.model, image, target, origin_img)

            val_num_img += val_batch_size
            val_sum_acc += val_batch_acc
            val_sum_dice += val_batch_dice
            val_sum_jacc += val_batch_jacc
            val_sum_sensitivity += val_batch_sensitivity
            val_sum_specificity += val_batch_specificity

            val_avg_acc = val_sum_acc / val_num_img
            val_avg_dice = val_sum_dice / val_num_img
            val_avg_jacc = val_sum_jacc / val_num_img
            val_avg_sensitivity = val_sum_sensitivity / val_num_img
            val_avg_specificity = val_sum_specificity / val_num_img
            val_tbar.set_description(
                'Validation: JA: {0:.4f}, DI: {1:.4f}, SE: {2:.4f}, SP: {3:.4f}, AC: {4:.4f}, img_num: {5}'.format(
                    val_avg_jacc, val_avg_dice, val_avg_sensitivity, val_avg_specificity, val_avg_acc, val_num_img))

        logger(self.log_file,
               'Validation: JA: {0:.4f}, DI: {1:.4f}, SE: {2:.4f}, SP: {3:.4f}, AC: {4:.4f}, img_num: {5}'.format(
                   val_avg_jacc, val_avg_dice, val_avg_sensitivity, val_avg_specificity, val_avg_acc, val_num_img))

        # new_pred = (test_avg_jacc + test_avg_dice) / 2
        new_pred = test_avg_jacc
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            checkpoint_name = 'fineturned_checkpoint.pth.tar' if self.args.ft else 'checkpoint.pth.tar'
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, self.args, False, checkpoint_name)
            print('best checkpoint saved !!!\n')
            logger(self.log_file, 'best checkpoint saved !!!\n')


def parse_args():
    # Traning setting options
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--devices', type=int, nargs='+', default=None)
    parser.add_argument('--dataset', default='ISIC2017')
    parser.add_argument('--data-num', type=int, default=100)
    parser.add_argument('--model', default='exp21_gsasnet_aspp_depthwise_v5')
    # parser.add_argument('--resume-dir', type=str, default='cityscapes/gsasnet_model/gsasnet101', metavar='PATH',
    #                     help='path to latest checkpoint (default: cityscapes/gsasnet_model/gsasnet101)')
    parser.add_argument('--backbone', default='xception65')
    parser.add_argument('--checkname', default='exp-x_gsasnetnet65_depthwise_v5_1e-4_by_1')
    parser.add_argument('--dilated', type=ast.literal_eval, default=True)
    parser.add_argument('--deep-base', type=ast.literal_eval, default=False)
    parser.add_argument('--img-size', type=int, nargs='+', default=[256, 256])
    parser.add_argument('--multi-grid', type=ast.literal_eval, default=True)
    parser.add_argument('--multi-dilation', type=int, nargs='+', default=[4, 4, 4])
    parser.add_argument('--output-stride', type=int, default=8)
    parser.add_argument('--high-rates', type=int, nargs='+', default=[2, 4])
    parser.add_argument('--aspp-rates', type=int, nargs='+', default=[6, 12, 18])
    parser.add_argument('--spp-size', type=int, nargs='+', default=[1, 2, 3, 6])
    parser.add_argument('--agg-out-dim', type=int, default=128)
    parser.add_argument('--aspp-out-dim', type=int, default=512)
    parser.add_argument('--up-conv', type=int, default=1, choices=[0, 1, 3])
    parser.add_argument('--ensemble', type=ast.literal_eval, default=False)
    parser.add_argument('--dynamic-lweight', type=ast.literal_eval, default=False)
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'prelu'])
    parser.add_argument('--weight', type=float, nargs='+', default=None)
    parser.add_argument('--xentropy-weight', type=float, default=1.0)
    parser.add_argument('--mae-weight', type=float, default=0.)
    parser.add_argument('--dice-weight', type=float, default=1.0)
    parser.add_argument('--boundary-weight', type=float, default=0.0)
    parser.add_argument('--boundary-kernel', type=int, default=5)
    parser.add_argument('--aux-weight', type=float, default=0.5)
    parser.add_argument('--loss1-4', type=float, default=0.0)
    parser.add_argument('--lovasz-weight', type=float, default=0.0)
    parser.add_argument('--lov1-4', type=float, default=0.0)
    parser.add_argument('--boundary-rf', type=ast.literal_eval, default=False)
    parser.add_argument('--refine-ce-weight', type=float, default=0.0)
    parser.add_argument('--refine-dice-weight', type=float, default=0.0)
    parser.add_argument('--margin-weight', type=float, default=0.0)
    parser.add_argument('--margin', type=float, default=0.0, help='the margin in margin loss.')
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
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--test-batch-size', type=int, default=16)
    parser.add_argument('--lr-scheduler', type=str, default='v1', choices=['v1', 'v2'])
    parser.add_argument('--head-lr-factor', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default='poly')
    # parser.add_argument('--lr-step', type=int, default=30)
    parser.add_argument('--se-loss', type=bool, default=False)
    parser.add_argument('--aux', type=ast.literal_eval, default=True)
    parser.add_argument('--resaux', type=ast.literal_eval, default=True)
    parser.add_argument('--aggaux', type=ast.literal_eval, default=False)
    parser.add_argument('--aux-fn', type=str, default='dice', choices=['bce', 'dice', 'lovasz', 'bce+dice'])
    parser.add_argument('--momentum', type=float, default='0.9', metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', type=float, default=1e-4, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--pretrained', type=ast.literal_eval, default=True,
                        help='whether to use pretrained base model')
    parser.add_argument('--refine', type=ast.literal_eval, default=False)
    parser.add_argument('--refine-ver', type=str, default='v2', choices=['v1', 'v2'])
    parser.add_argument('--freezn', type=ast.literal_eval, default=False)
    parser.add_argument('--pretrained-file', type=str, default=None, help='resnet101-2a57e44d.pth')
    parser.add_argument('--no-val', type=bool, default=False,
                        help='whether not using validation (default: False)')
    parser.add_argument('--ft', type=ast.literal_eval, default=False,
                        help='whether to fine turning (default: True for training)')
    parser.add_argument('--log-file', type=str, default='log.txt')
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
    trainer = Trainer(args)
    print('Starting Epoch: {}'.format(args.start_epoch))
    logger(args.resume_dir + '/' + args.log_file, 'Starting Epoch: {}'.format(args.start_epoch))
    print('Total Epochs: {}'.format(args.epochs))
    logger(args.resume_dir + '/' + args.log_file, 'Total Epochs: {}'.format(args.epochs))
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        if not args.no_val:
            trainer.validation_and_test(epoch)
        torch.cuda.empty_cache()
