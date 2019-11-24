import logging
import math
from collections import OrderedDict

import numpy as np

import config as cfg
import tools
import torch
import torch.nn as nn

# from model.backbone.mobilenet import mobilenetv3_large 
# from model.backbone.mobilenetv3 import MobileNetV3_Large
# from model.backbone.mobilenetv2 import MobileNetV2
from model.backbone.mobilev2 import MobileNetV2
import sys


class conv_bn(nn.Module):
    def __init__(self,inp, oup, kernel,stride,padding):
        super(conv_bn, self).__init__()
        self.convbn=nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(inp, oup, kernel, stride, padding, bias=False)),
            ('bn', nn.BatchNorm2d(oup)),
            ('relu', nn.ReLU6(inplace=True))
        ]))
    def forward(self, input):
        return self.convbn(input)

class conv_bias(nn.Module):
    def __init__(self,inp, oup, kernel,stride,padding):
        super(conv_bias, self).__init__()
        self.conv=nn.Conv2d(inp, oup, kernel, stride, padding, bias=True)
    def forward(self, input):
        return self.conv(input)

class sepconv_bn(nn.Module):
    def __init__(self,inp, oup, kernel,stride,padding):
        super(sepconv_bn, self).__init__()
        self.sepconv_bn= nn.Sequential(OrderedDict([
            ('sepconv',nn.Conv2d(inp, inp, kernel, stride, padding,groups=inp, bias=False)),
            ('sepbn',nn.BatchNorm2d(inp)),
            ('seprelu',nn.ReLU6(inplace=True)),
            ('pointconv', nn.Conv2d(inp, oup, 1, 1, 0, bias=False)),
            ('pointbn', nn.BatchNorm2d(oup)),
            ('pointrelu', nn.ReLU6(inplace=True)),
        ]))
    def forward(self, input):
        return self.sepconv_bn(input)


class Decode(nn.Module):
    def __init__(self, num_classes):
        super(Decode, self).__init__()
        self.__num_classes = num_classes

    def forward(self, conv, stride):
        conv = conv.permute(0, 2, 3, 1)
        conv_shape = conv.size()
        batch_size = conv_shape[0]
        out_size = conv_shape[1]
        gt_per_grid = conv_shape[3] // (5 + self.__num_classes)

        conv = conv.view(batch_size, out_size, out_size, gt_per_grid, 5 + self.__num_classes)
        conv_raw_dx1dy1,conv_raw_dx2dy2,conv_raw_conf,conv_raw_prob=torch.split(conv,[2,2,1,self.__num_classes],dim=4)

        shiftx=torch.arange(0,out_size,dtype=torch.float32)
        shifty=torch.arange(0,out_size,dtype=torch.float32)
        shifty,shiftx=torch.meshgrid([shiftx,shifty])
        shiftx=shiftx.unsqueeze(-1).repeat(batch_size,1,1,3)
        shifty=shifty.unsqueeze(-1).repeat(batch_size,1,1,3)
        xy_grid=torch.stack([shiftx,shifty],dim=4).cuda()

        # decode xy
        pred_xymin = (xy_grid + 0.5 - torch.exp(conv_raw_dx1dy1)) * stride
        pred_xymax = (xy_grid + 0.5 + torch.exp(conv_raw_dx2dy2)) * stride
        pred_xy = torch.cat((pred_xymin, pred_xymax), -1)
        # decode confidence
        pred_conf = torch.sigmoid(conv_raw_conf)
        # decode probability
        pred_prob = torch.sigmoid(conv_raw_prob)
        # re-concat decoded items
        pred_bbox = torch.cat((pred_xy, pred_conf, pred_prob), -1)
        return pred_bbox


class YOLOv3(nn.Module):
    def __init__(self):
        super(YOLOv3, self).__init__()

        self.__classes = cfg.CLASSES
        self.__num_classes = len(cfg.CLASSES)
        self.__strides = np.array(cfg.STRIDES)
        self.__gt_per_grid = cfg.GT_PER_GRID
        self.__iou_loss_thresh = cfg.IOU_LOSS_THRESH

        self.backbone = MobileNetV2()
        self.__backbone_weights = cfg.BACKBONE_WEIGHTS
        out_filters = self.backbone.layers_out_filters
        yolo_filters = self.__gt_per_grid * (5 + self.__num_classes)
        self.decode = Decode(self.__num_classes)

        self.heads=[]
        self.headslarge=nn.Sequential(OrderedDict([
            ('conv0',conv_bn(out_filters[2],512,kernel=1,stride=1,padding=0)),
            ('conv1', sepconv_bn(512, 1024, kernel=3, stride=1, padding=1)),
            ('conv2', conv_bn(1024, 512, kernel=1,stride=1,padding=0)),
            ('conv3', sepconv_bn(512, 1024, kernel=3, stride=1, padding=1)),
            ('conv4', conv_bn(1024, 512, kernel=1,stride=1,padding=0)),
        ]))
        self.detlarge=nn.Sequential(OrderedDict([
            ('conv5',sepconv_bn(512,1024,kernel=3, stride=1, padding=1)),
            ('conv6', conv_bias(1024, yolo_filters,kernel=1,stride=1,padding=0))
        ]))
        self.mergelarge=nn.Sequential(OrderedDict([
            ('conv7',conv_bn(512,256,kernel=1,stride=1,padding=0)),
            ('upsample0',nn.UpsamplingNearest2d(scale_factor=2)),
        ]))
        #-----------------------------------------------
        self.headsmid=nn.Sequential(OrderedDict([
            ('conv8',conv_bn(out_filters[1]+256,256,kernel=1,stride=1,padding=0)),
            ('conv9', sepconv_bn(256, 512, kernel=3, stride=1, padding=1)),
            ('conv10', conv_bn(512, 256, kernel=1,stride=1,padding=0)),
            ('conv11', sepconv_bn(256, 512, kernel=3, stride=1, padding=1)),
            ('conv12', conv_bn(512, 256, kernel=1,stride=1,padding=0)),
        ]))
        self.detmid=nn.Sequential(OrderedDict([
            ('conv13',sepconv_bn(256,512,kernel=3, stride=1, padding=1)),
            ('conv14', conv_bias(512, yolo_filters,kernel=1,stride=1,padding=0))
        ]))
        self.mergemid=nn.Sequential(OrderedDict([
            ('conv15',conv_bn(256,128,kernel=1,stride=1,padding=0)),
            ('upsample0',nn.UpsamplingNearest2d(scale_factor=2)),
        ]))
        #-----------------------------------------------
        self.headsmall=nn.Sequential(OrderedDict([
            ('conv16',conv_bn(out_filters[0]+128,128,kernel=1,stride=1,padding=0)),
            ('conv17', sepconv_bn(128, 256, kernel=3, stride=1, padding=1)),
            ('conv18', conv_bn(256, 128, kernel=1,stride=1,padding=0)),
            ('conv19', sepconv_bn(128, 256, kernel=3, stride=1, padding=1)),
            ('conv20', conv_bn(256, 128, kernel=1,stride=1,padding=0)),
        ]))
        self.detsmall=nn.Sequential(OrderedDict([
            ('conv21',sepconv_bn(128,256,kernel=3, stride=1, padding=1)),
            ('conv22', conv_bias(256, yolo_filters,kernel=1,stride=1,padding=0))
        ]))

    def loss_per_scale(self, conv, pred, label, bboxes, stride):

        def __focal(target, actual, alpha=1, gamma=2):
            focal = alpha * torch.pow(torch.abs(target - actual), gamma)
            return focal

        bce_wl_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        smooth_loss = torch.nn.SmoothL1Loss(reduction='none')

        conv = conv.permute(0, 2, 3, 1)
        conv_shape = conv.size()
        batch_size = conv_shape[0]
        out_size = conv_shape[1]
        in_size = stride * out_size
        conv = conv.view(batch_size, out_size, out_size, self.__gt_per_grid, 5 + self.__num_classes)
        conv_raw_conf = conv[..., 4:5]
        conv_raw_prob = conv[..., 5:]

        pred_coor = pred[..., 0:4]
        pred_conf = pred[..., 4:5]

        label_coor = label[..., 0:4]
        respond_bbox = label[..., 4:5]
        label_prob = label[..., 5:-1]
        label_mixw = label[..., -1:]

        bbox_wh = label_coor[..., 2:] - label_coor[..., :2]
        bbox_loss_scale = 2.0 - 1.0 * bbox_wh[..., 0:1] * bbox_wh[..., 1:2] / (in_size ** 2)
        # l1 loss
        # l1_loss = respond_bbox * bbox_loss_scale * smooth_loss(target=label_coor, input=pred_coor) * 0.1

        # giou loss
        giou = tools.giou(pred_coor, label_coor)
        giou = giou[..., None]
        giou_loss = respond_bbox * bbox_loss_scale * (1.0 - giou)

        # confidence loss
        iou = tools.iou_calc3(pred_coor[:, :, :, :, None, :],
            bboxes[:, None, None, None, :, : ])
        max_iou, _ = torch.max(iou, -1)
        max_iou = max_iou[..., None]
        respond_bgd = (1.0 - respond_bbox) * (max_iou < self.__iou_loss_thresh).float()
        # print('max', (max_iou >= self.__iou_loss_thresh).float().sum([1, 2, 3, 4]).mean().item())

        conf_focal = __focal(respond_bbox, pred_conf)

        conf_loss = conf_focal * (
            respond_bbox * bce_wl_loss(conv_raw_conf, respond_bbox)
            +
            respond_bgd * bce_wl_loss(conv_raw_conf, respond_bbox)
        )

        # classes loss
        prob_loss = respond_bbox * bce_wl_loss(conv_raw_prob, label_prob)

        # sum up
        loss = torch.cat([giou_loss, conf_loss, prob_loss], -1)
        loss = loss * label_mixw
        loss = loss.sum([1, 2, 3, 4]).mean()

        # print('bgd', respond_bgd.sum([1, 2, 3, 4]).mean().item())
        if torch.isnan(loss):
            # print(conv_raw_conf.mean())
            print(giou_loss.sum([1, 2, 3, 4]).mean())
            print(conf_loss.sum([1, 2, 3, 4]).mean())
            print(prob_loss.sum([1, 2, 3, 4]).mean())
            sys.exit(0)
        return loss, (conf_loss*label_mixw).sum([1, 2, 3, 4]).mean()

    def forward(self, x, target=None):
        backbone_small, backbone_middle, backbone_large = self.backbone(x)
        conv=self.headslarge(backbone_large)
        outlarge=self.detlarge(conv)

        conv=self.mergelarge(conv)
        conv=self.headsmid(torch.cat((conv,backbone_middle),dim=1))
        outmid=self.detmid(conv)

        conv=self.mergemid(conv)

        conv=self.headsmall(torch.cat((conv,backbone_small),dim=1))
        outsmall=self.detsmall(conv)
        predlarge = self.decode(outlarge, 32)
        predmid = self.decode(outmid,16)
        predsmall = self.decode(outsmall,8)
        if target is None:
            return predsmall, predmid, predlarge
        
        label_sbbox, label_mbbox, label_lbbox,\
            sbbox, mbbox, lbbox = target
        loss_sbbox, cl1 = self.loss_per_scale(outsmall, predsmall,
            label_sbbox, sbbox, self.__strides[0])
        loss_mbbox, cl2 = self.loss_per_scale(outmid, predmid,
            label_mbbox, mbbox, self.__strides[1])
        loss_lbbox, cl3 = self.loss_per_scale(outlarge, predlarge,
            label_lbbox, lbbox, self.__strides[2])
        return loss_sbbox + loss_mbbox + loss_lbbox, cl1 + cl2 + cl3


    def init_weights(self):
        # def _init_weights(m):
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #             # m.bias.data[:4].fill_(-4.)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1.)
        #         m.bias.data.zero_()
        # self.apply(_init_weights)
        if self.__backbone_weights is not None:
            print('Loading backbone weights from {}'.format(self.__backbone_weights))
            self.backbone.load_weights(self.__backbone_weights)
