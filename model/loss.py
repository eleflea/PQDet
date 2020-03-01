import sys

import torch
from torch import nn

import tools


def loss_per_scale(pred, label, bboxes, stride, ignore_thresh):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor, int, float) -> Tuple[torch.Tensor]

    def __focal(target, actual, alpha=1, gamma=2):
        focal = alpha * torch.pow(torch.abs(target - actual), gamma)
        return focal

    bce_loss = nn.BCELoss(reduction='none')
    out_size_h, out_size_w = pred.shape[1:3]
    in_size = (stride * out_size_h, stride * out_size_w)

    pred_coor = pred[..., 0:4]
    pred_conf = pred[..., 4:5]
    pred_prob = pred[..., 5:]

    label_coor = label[..., 0:4]
    respond_bbox = label[..., 4:5]
    label_prob = label[..., 5:-1]
    label_mixw = label[..., -1:]

    bbox_wh = label_coor[..., 2:] - label_coor[..., :2]
    bbox_loss_scale = 2.0 - 1.0 * \
        bbox_wh[..., 0:1] * bbox_wh[..., 1:2] / (in_size[0] * in_size[1])
    # l1 loss
    # l1_loss = respond_bbox * bbox_loss_scale * smooth_loss(target=label_coor, input=pred_coor) * 0.1

    # giou loss
    giou = tools.giou(pred_coor, label_coor)
    giou = giou[..., None]
    giou_loss = respond_bbox * bbox_loss_scale * (1.0 - giou)

    # confidence loss
    iou = tools.iou_calc3(pred_coor[:, :, :, :, None, :],
                            bboxes[:, None, None, None, :, :])
    max_iou, _ = torch.max(iou, -1)
    max_iou = max_iou[..., None]
    respond_bgd = (1.0 - respond_bbox) * \
        (max_iou < ignore_thresh).float()
    # print('max', (max_iou >= ignore_thresh).float().sum([1, 2, 3, 4]).mean().item())

    conf_focal = __focal(respond_bbox, pred_conf)

    conf_loss = conf_focal * (
        respond_bbox * bce_loss(pred_conf, respond_bbox)
        +
        respond_bgd * bce_loss(pred_conf, respond_bbox)
    )

    # classes loss
    prob_loss = respond_bbox * bce_loss(pred_prob, label_prob)

    # sum up
    #loss = torch.cat([giou_loss, conf_loss, prob_loss], -1)
    giou_loss = (giou_loss * label_mixw).sum([1, 2, 3, 4]).mean()
    conf_loss = (conf_loss * label_mixw).sum([1, 2, 3, 4]).mean()
    prob_loss = (prob_loss * label_mixw).sum([1, 2, 3, 4]).mean()
    loss = giou_loss + conf_loss + prob_loss

    # print('bgd', respond_bgd.sum([1, 2, 3, 4]).mean().item())
    if torch.isnan(loss):
        # print(conv_raw_conf.mean())
        print(giou_loss)
        print(conf_loss)
        print(prob_loss)
        sys.exit(0)
    return loss, giou_loss, conf_loss, prob_loss
