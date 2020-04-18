##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Encoding Custermized NN Module"""
import torch
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, MSELoss, Sigmoid, L1Loss
from torch.nn import functional as F
from torch.autograd import Variable


torch_ver = torch.__version__[:3]

__all__ = ['GramMatrix', 'SegmentationLosses', 'View', 'Sum', 'Mean',
           'Normalize', 'PyramidPooling', 'SegmentationMultiLosses']


class GramMatrix(Module):
    r""" Gram Matrix for a 4D convolutional featuremaps as a mini-batch

    .. math::
        \mathcal{G} = \sum_{h=1}^{H_i}\sum_{w=1}^{W_i} \mathcal{F}_{h,w}\mathcal{F}_{h,w}^T
    """

    def forward(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram


def softmax_crossentropy(input, target, weight, size_average, ignore_index, reduce=True):
    return F.nll_loss(F.log_softmax(input, 1), target, weight,
                      size_average, ignore_index, reduce)


class SegmentationLosses(CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""

    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1, aux=False, aux_weight=0.5, dice_weight=0.,
                 xentropy_weight=1., loss1_4=1.0, lov1_4=1.0, mae_weight=0., lovasz_weight=0.5, margin_weight=0.,
                 refine_ce_weight=0.0, refine_dice_weight=1.0, weight=None, margin=0.1, k=30, c=5, ensemble=False,
                 resaux=False, aggaux=False, aux_fn='bce', boundary_kernel=5, boundary_weight=0.0, refine=False,
                 boundary_rf=False, size_average=None, ignore_index=-1):
        super(SegmentationLosses, self).__init__(weight, size_average, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.resaux = resaux
        self.aggaux = aggaux
        self.aux_fn = aux_fn
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.dice_weight = dice_weight
        self.xentropy_weight = xentropy_weight
        self.boundary_weight = boundary_weight
        self.mae_weight = mae_weight
        self.lovasz_weight = lovasz_weight
        self.loss1_4 = loss1_4
        self.lov1_4 = lov1_4
        self.margin_weight = margin_weight
        self.refine_ce_weight = refine_ce_weight
        self.refine_dice_weight = refine_dice_weight
        self.ensemble = ensemble
        self.refine = refine
        self.boundary_rf = boundary_rf
        self.boundary_kernel = boundary_kernel
        self.bceloss = BCELoss(weight, size_average)
        self.mseloss = MSELoss()
        self.maeloss = L1Loss()
        self.diceloss_with_sigmoid = DiceLoss_with_sigmoid()
        self.diceloss_with_softmax = DiceLoss_with_softmax()
        self.sigmoid = Sigmoid()
        self.margin_loss = Margin_Loss(margin=margin, k=k, c=c)
        self.boundary_loss = Boundary_loss(kernel_size=boundary_kernel)

    def forward(self, *inputs):
        # print(inputs[0].size())
        if not self.se_loss and not self.aux:
            if inputs[0] == 6:
                (pred, p0, l1, l2, l3, l4), target = inputs
                # xentropy_loss = binary_xloss(pred, target.float())
                xentropy_loss = super(SegmentationLosses, self).forward(pred, target)
                if self.ensemble:
                    loss0 = super(SegmentationLosses, self).forward(p0, target)
                    lovasz0 = lovasz_softmax(p0.softmax(1), target)
                loss1 = super(SegmentationLosses, self).forward(l1, target)
                lovasz1 = lovasz_softmax(l1.softmax(1), target, [1])
                loss2 = super(SegmentationLosses, self).forward(l2, target)
                lovasz2 = lovasz_softmax(l2.softmax(1), target, [1])
                loss3 = super(SegmentationLosses, self).forward(l3, target)
                lovasz3 = lovasz_softmax(l3.softmax(1), target, [1])
                loss4 = super(SegmentationLosses, self).forward(l4, target)
                lovasz4 = lovasz_softmax(l4.softmax(1), target, [1])
                mae_loss = self.maeloss(pred.softmax(dim=1)[:, 1], target.float())
                # dice_loss = self.diceloss_with_sigmoid(pred, target.float())
                dice_loss = self.diceloss_with_softmax(pred, target.float())
                # lovasz_loss = lovasz_hinge(pred, target)
                lovasz_loss = lovasz_softmax(pred.softmax(1), target, [1])
                m_loss = self.margin_loss(pred, target.float(), pred)
                b_loss = self.boundary_loss(pred, target.float())
                # mae_loss = 0
                # dice_loss = 0
                # lovasz_loss = 0
                # m_loss = 0
                if self.ensemble:
                    return self.xentropy_weight * xentropy_loss + self.boundary_weight * b_loss + self.self.loss1_4 * (
                            loss1 + loss2 + loss3 + loss4 + loss0) + self.lov1_4 * (
                                   lovasz1 + lovasz2 + lovasz3 + lovasz4 + lovasz0) + self.mae_weight * mae_loss + self.dice_weight * dice_loss + self.lovasz_weight * lovasz_loss + self.margin_weight * m_loss, xentropy_loss, b_loss, loss1, lovasz1, loss2, lovasz2, loss3, lovasz3, loss4, lovasz4, loss0, lovasz0, mae_loss, dice_loss, lovasz_loss, m_loss
                else:
                    return self.xentropy_weight * xentropy_loss + self.boundary_weight * b_loss + self.loss1_4 * (
                            loss1 + loss2 + loss3 + loss4) + self.lov1_4 * (
                                   lovasz1 + lovasz2 + lovasz3 + lovasz4) + self.mae_weight * mae_loss + self.dice_weight * dice_loss + self.lovasz_weight * lovasz_loss + self.margin_weight * m_loss, xentropy_loss, b_loss, loss1, lovasz1, loss2, lovasz2, loss3, lovasz3, loss4, lovasz4, mae_loss, dice_loss, lovasz_loss, m_loss
            elif inputs[0] == 5:
                (pred, p0, l1, l2, l3), target = inputs
                # xentropy_loss = binary_xloss(pred, target.float())
                xentropy_loss = super(SegmentationLosses, self).forward(pred, target)
                if self.ensemble:
                    loss0 = super(SegmentationLosses, self).forward(p0, target)
                    lovasz0 = lovasz_softmax(p0.softmax(1), target, [1])
                loss1 = super(SegmentationLosses, self).forward(l1, target)
                lovasz1 = lovasz_softmax(l1.softmax(1), target, [1])
                loss2 = super(SegmentationLosses, self).forward(l2, target)
                lovasz2 = lovasz_softmax(l2.softmax(1), target, [1])
                loss3 = super(SegmentationLosses, self).forward(l3, target)
                lovasz3 = lovasz_softmax(l3.softmax(1), target, [1])
                mae_loss = self.maeloss(pred.softmax(dim=1)[:, 1], target.float())
                # dice_loss = self.diceloss_with_sigmoid(pred, target.float())
                dice_loss = self.diceloss_with_softmax(pred, target.float())
                # lovasz_loss = lovasz_hinge(pred, target)
                lovasz_loss = lovasz_softmax(pred.softmax(1), target, [1])
                m_loss = self.margin_loss(pred, target.float(), pred)
                b_loss = self.boundary_loss(pred, target.float())
                # mae_loss = 0
                # dice_loss = 0
                # lovasz_loss = 0
                # m_loss = 0
                if self.ensemble:
                    return self.xentropy_weight * xentropy_loss + self.boundary_weight * b_loss + self.loss1_4 * (
                            loss1 + loss2 + loss3 + loss0) + self.lov1_4 * (
                                   lovasz1 + lovasz2 + lovasz3 + lovasz0) + self.mae_weight * mae_loss + self.dice_weight * dice_loss + self.lovasz_weight * lovasz_loss + self.margin_weight * m_loss, xentropy_loss, b_loss, loss1, lovasz1, loss2, lovasz2, loss3, lovasz3, torch.zeros(
                        1).cuda(), torch.zeros(1).cuda(), loss0, lovasz0, mae_loss, dice_loss, lovasz_loss, m_loss
                else:
                    return self.xentropy_weight * xentropy_loss + self.boundary_weight * b_loss + self.loss1_4 * (
                            loss1 + loss2 + loss3) + self.lov1_4 * (
                                   lovasz1 + lovasz2 + lovasz3) + self.mae_weight * mae_loss + self.dice_weight * dice_loss + self.lovasz_weight * lovasz_loss + self.margin_weight * m_loss, xentropy_loss, b_loss, loss1, lovasz1, loss2, lovasz2, loss3, lovasz3, torch.zeros(
                        1).cuda(), torch.zeros(1).cuda(), mae_loss, dice_loss, lovasz_loss, m_loss
        elif not self.se_loss:
            if not self.resaux and not self.aggaux:
                if self.refine:
                    (pred, p0, refine_p), target = inputs
                else:
                    (pred, p0), target = inputs
                xentropy_loss = super(SegmentationLosses, self).forward(pred, target)
                loss1 = torch.zeros(1).cuda()
                lovasz1 = torch.zeros(1).cuda()
                loss2 = torch.zeros(1).cuda()
                lovasz2 = torch.zeros(1).cuda()
            elif not self.aggaux:
                if self.refine:
                    (pred, p0, l1, refine_p), target = inputs
                else:
                    (pred, p0, l1), target = inputs
                # xentropy_loss = binary_xloss(pred, target.float())
                xentropy_loss = super(SegmentationLosses, self).forward(pred, target)
                if self.aux_fn == 'dice':
                    loss1 = self.diceloss_with_softmax(l1, target.float())
                elif self.aux_fn == 'bce':
                    loss1 = super(SegmentationLosses, self).forward(l1, target)
                elif self.aux_fn == 'lovasz':
                    loss1 = lovasz_softmax(l1.softmax(1), target, [1])
                elif self.aux_fn == 'bce+dice':
                    loss1 = super(SegmentationLosses, self).forward(l1, target) + lovasz_softmax(l1.softmax(1), target,
                                                                                                 [1])
                lovasz1 = lovasz_softmax(l1.softmax(1), target, [1])
                loss2 = torch.zeros(1).cuda()
                lovasz2 = torch.zeros(1).cuda()
            elif not self.resaux:
                if self.refine:
                    (pred, p0, l1, refine_p), target = inputs
                else:
                    (pred, p0, l1), target = inputs
                # xentropy_loss = binary_xloss(pred, target.float())
                xentropy_loss = super(SegmentationLosses, self).forward(pred, target)
                if self.aux_fn == 'dice':
                    loss1 = self.diceloss_with_softmax(l1, target.float())
                elif self.aux_fn == 'bce':
                    loss1 = super(SegmentationLosses, self).forward(l1, target)
                elif self.aux_fn == 'lovasz':
                    loss1 = lovasz_softmax(l1.softmax(1), target, [1])
                elif self.aux_fn == 'bce+dice':
                    loss1 = super(SegmentationLosses, self).forward(l1, target) + lovasz_softmax(l1.softmax(1), target,
                                                                                                 [1])
                lovasz1 = lovasz_softmax(l1.softmax(1), target, [1])
                loss2 = torch.zeros(1).cuda()
                lovasz2 = torch.zeros(1).cuda()
            else:
                if self.refine:
                    (pred, p0, l1, l2, refine_p), target = inputs
                else:
                    (pred, p0, l1, l2), target = inputs
                # xentropy_loss = binary_xloss(pred, target.float())
                xentropy_loss = super(SegmentationLosses, self).forward(pred, target)
                if self.aux_fn == 'dice':
                    loss1 = self.diceloss_with_softmax(l1, target.float())
                    loss2 = self.diceloss_with_softmax(l2, target.float())
                elif self.aux_fn == 'bce':
                    loss1 = super(SegmentationLosses, self).forward(l1, target)
                    loss2 = super(SegmentationLosses, self).forward(l2, target)
                elif self.aux_fn == 'lovasz':
                    loss1 = lovasz_softmax(l1.softmax(1), target, [1])
                    loss2 = lovasz_softmax(l2.softmax(1), target, [1])
                elif self.aux_fn == 'bce+dice':
                    loss1 = super(SegmentationLosses, self).forward(l1, target) + lovasz_softmax(l1.softmax(1), target,
                                                                                                 [1])
                    loss2 = super(SegmentationLosses, self).forward(l2, target) + lovasz_softmax(l2.softmax(1), target,
                                                                                                 [1])
                lovasz1 = lovasz_softmax(l1.softmax(1), target, [1])
                lovasz2 = lovasz_softmax(l2.softmax(1), target, [1])
            if self.ensemble:
                loss0 = super(SegmentationLosses, self).forward(p0, target)
                lovasz0 = lovasz_softmax(p0.softmax(1), target, [1])
            pred_tuple, _ = inputs
            pred = pred_tuple[0]
            if self.refine:
                refine_p = pred_tuple[-1]
            loss3 = torch.zeros(1).cuda()
            lovasz3 = torch.zeros(1).cuda()
            loss4 = torch.zeros(1).cuda()
            lovasz4 = torch.zeros(1).cuda()
            mae_loss = self.maeloss(pred.softmax(dim=1)[:, 1], target.float())
            # dice_loss = self.diceloss_with_sigmoid(pred, target.float())
            dice_loss = self.diceloss_with_softmax(pred, target.float())
            # lovasz_loss = lovasz_hinge(pred, target)
            lovasz_loss = lovasz_softmax(pred.softmax(1), target, [1])
            m_loss = self.margin_loss(pred, target.float(), pred)
            b_loss = self.boundary_loss(pred, target.float())
            if self.refine:
                if self.boundary_rf:
                    boundary_target = F.avg_pool2d(target.float(), self.boundary_kernel, stride=1,
                                                   padding=(self.boundary_kernel - 1) // 2)
                    refine_xentropy = self.maeloss(refine_p.softmax(1)[:, 1],
                                                   torch.abs(target.float() - boundary_target))
                    refine_dice_loss = torch.zeros(1).cuda()
                else:
                    refine_xentropy = super(SegmentationLosses, self).forward(refine_p, target)
                    refine_dice_loss = self.diceloss_with_softmax(refine_p, target.float())
            if self.ensemble:
                total_loss = self.xentropy_weight * xentropy_loss + self.boundary_weight * b_loss + self.aux_weight * (
                        loss1 + loss2 + loss3 + loss4 + loss0) + self.lov1_4 * (
                                     lovasz1 + lovasz2 + lovasz3 + lovasz4 + lovasz0) + self.mae_weight * mae_loss + self.dice_weight * dice_loss + self.lovasz_weight * lovasz_loss + self.margin_weight * m_loss
                if self.refine:
                    total_loss += (self.refine_ce_weight * refine_xentropy + self.refine_dice_weight * refine_dice_loss)
                    return total_loss, xentropy_loss, b_loss, loss1, lovasz1, loss2, lovasz2, loss3, lovasz3, loss4, lovasz4, loss0, lovasz0, mae_loss, dice_loss, lovasz_loss, m_loss, refine_xentropy, refine_dice_loss
                else:
                    return total_loss, xentropy_loss, b_loss, loss1, lovasz1, loss2, lovasz2, loss3, lovasz3, loss4, lovasz4, loss0, lovasz0, mae_loss, dice_loss, lovasz_loss, m_loss
            else:
                total_loss = self.xentropy_weight * xentropy_loss + self.boundary_weight * b_loss + self.aux_weight * (
                        loss1 + loss2 + loss3) + self.lov1_4 * (
                                     lovasz1 + lovasz2 + lovasz3) + self.mae_weight * mae_loss + self.dice_weight * dice_loss + self.lovasz_weight * lovasz_loss + self.margin_weight * m_loss
                if self.refine:
                    total_loss += (self.refine_ce_weight * refine_xentropy + self.refine_dice_weight * refine_dice_loss)
                    return total_loss, xentropy_loss, b_loss, loss1, lovasz1, loss2, lovasz2, loss3, lovasz3, loss4, lovasz4, mae_loss, dice_loss, lovasz_loss, m_loss, refine_xentropy, refine_dice_loss
                else:
                    return total_loss, xentropy_loss, b_loss, loss1, lovasz1, loss2, lovasz2, loss3, lovasz3, loss4, lovasz4, mae_loss, dice_loss, lovasz_loss, m_loss
        elif not self.aux:
            (pred, se_pred), target = inputs
            # print('pred: {}\nse_pred: {}\ntarget: {}'.format(pred.size(), se_pred.size(), target.size()))
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(SegmentationLosses, self).forward(pred, target)
            loss2 = self.bceloss(F.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2
        else:
            (pred1, se_pred, pred2), target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            loss3 = self.bceloss(F.sigmoid(se_pred), se_target)
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(),
                               bins=nclass, min=0,
                               max=nclass - 1)
            vect = hist > 0
            tvect[i] = vect
        return tvect


class DiceLoss_with_sigmoid(Module):
    def __init__(self):
        super(DiceLoss_with_sigmoid, self).__init__()

    def forward(self, predict, target):
        predict = F.sigmoid(predict)
        predict[predict >= 0.5] = 1
        predict[predict != 1] = 0
        N = target.size(0)
        smooth = 1

        predict_flat = predict.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = predict_flat * target_flat

        loss = 2. * (intersection.sum(1) + smooth) / (predict_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class DiceLoss_with_softmax(Module):
    def __init__(self):
        super(DiceLoss_with_softmax, self).__init__()

    def forward(self, predict, target):
        predict = F.softmax(predict, dim=1)
        # predict = predict.max(dim=1)[1].float()
        predict = predict[:, 1]
        N = target.size(0)
        smooth = 1

        predict_flat = predict.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = predict_flat * target_flat

        loss = 2. * (intersection.sum(1) + smooth) / (predict_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class SegmentationMultiLosses(CrossEntropyLoss):
    """2D Cross Entropy Loss with Multi-L1oss"""

    def __init__(self, nclass=-1, weight=None, size_average=True, ignore_index=-1):
        super(SegmentationMultiLosses, self).__init__(weight, size_average, ignore_index)
        self.nclass = nclass

    def forward(self, *inputs):
        (pred1, pred2, pred3), target = tuple(inputs)
        # pred1, pred2, pred3 = tuple(preds)

        loss1 = super(SegmentationMultiLosses, self).forward(pred1, target)
        loss2 = super(SegmentationMultiLosses, self).forward(pred2, target)
        loss3 = super(SegmentationMultiLosses, self).forward(pred3, target)
        loss = loss1 + loss2 + loss3
        return loss


class View(Module):
    """Reshape the input into different size, an inplace operator, support
    SelfParallel mode.
    """

    def __init__(self, *args):
        super(View, self).__init__()
        if len(args) == 1 and isinstance(args[0], torch.Size):
            self.size = args[0]
        else:
            self.size = torch.Size(args)

    def forward(self, input):
        return input.view(self.size)


class Sum(Module):
    def __init__(self, dim, keep_dim=False):
        super(Sum, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.sum(self.dim, self.keep_dim)


class Mean(Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)


class Normalize(Module):
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.

    Does:

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}

    for each subtensor v over dimension dim of input. Each subtensor is
    flattened into a vector, i.e. :math:`\lVert v \rVert_p` is not a matrix
    norm.

    With default arguments normalizes over the second dimension with Euclidean
    norm.

    Args:
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
    """

    def __init__(self, p=2, dim=1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, self.p, self.dim, eps=1e-8)


class PyramidPooling(Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, in_channels, norm_layer, up_kwargs, spp_size=[1, 2, 3, 6]):
        super(PyramidPooling, self).__init__()
        self.pool1 = AdaptiveAvgPool2d(spp_size[0])
        self.pool2 = AdaptiveAvgPool2d(spp_size[1])
        self.pool3 = AdaptiveAvgPool2d(spp_size[2])
        self.pool4 = AdaptiveAvgPool2d(spp_size[3])

        out_channels = int(in_channels / 4)
        self.conv1 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv2 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv3 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv4 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        # bilinear upsample options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
        feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
        feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
        feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)
        return torch.cat((x, feat1, feat2, feat3, feat4), 1)
