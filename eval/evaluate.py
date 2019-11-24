import os
import shutil
from math import ceil

import cv2
import numpy as np
import torch
from tqdm import tqdm

import config as cfg
import tools
from dataset.dataset import get_sample_by_img_path
from eval import voc_eval


class Evaluator(object):

    def __init__(self, model):
        self._dataset_name = cfg.DATASET_NAME
        self._dataset_path = cfg.DATASET_PATH
        self._test_input_size = cfg.TEST_INPUT_SIZE
        self._test_dataset_file = cfg.TEST_DATASET_FILE
        self._classes = cfg.CLASSES
        self._num_classes = len(self._classes)
        self._class_to_ind = dict(zip(self._classes, range(self._num_classes)))
        self._score_threshold = cfg.SCORE_THRESHOLD
        self._iou_threshold = cfg.IOU_THRESHOLD
        self._map_iou = cfg.MAP_IOU
        self._test_batch_size = cfg.TEST_BATCH_SIZE

        cuda = torch.cuda.is_available()
        if cuda:
            model = model.cuda()
        self._dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        self.model = model

    def __prepare_img(self, img_paths):
        input_imgs = np.zeros((len(img_paths), 3, self._test_input_size, self._test_input_size), dtype=np.float32)
        img_shapes = []
        for i, img_path in enumerate(img_paths):
            img, _ = get_sample_by_img_path(img_path)
            org_h, org_w, _ = img.shape
            input_img = tools.img_preprocess2(img, None, (self._test_input_size, self._test_input_size), False)
            input_imgs[i, ...] = input_img
            img_shapes.append((org_h, org_w))
        return input_imgs, img_shapes

    def __perdict(self, images, images_info):
        bs = images.shape[0]
        with torch.no_grad():
            pred_sbbox, pred_mbbox, pred_lbbox = self.model(torch.from_numpy(images).cuda())

        pred_bboxs = torch.cat([pred_sbbox.view((bs, -1, 5 + self._num_classes)),
                                    pred_mbbox.view((bs, -1, 5 + self._num_classes)),
                                    pred_lbbox.view((bs, -1, 5 + self._num_classes))], 1)
        batch_bboxes = []
        for i, pred_bbox in enumerate(pred_bboxs):
            bboxes = self.__convert_pred(pred_bbox, self._test_input_size, images_info[i]).detach().cpu().numpy()
            bboxes = tools.nms(bboxes, self._score_threshold, self._iou_threshold, method='nms')
            batch_bboxes.append(bboxes)
        return batch_bboxes
    
    def __convert_pred(self, pred_bbox, test_input_size, org_img_shape):
        """
        将yolo输出的bbox信息(xmin, ymin, xmax, ymax, confidence, probability)进行转换，
        其中(xmin, ymin, xmax, ymax)是预测bbox的左上角和右下角坐标
        confidence是预测bbox属于物体的概率，probability是条件概率分布
        (xmin, ymin, xmax, ymax) --> (xmin_org, ymin_org, xmax_org, ymax_org)
        --> 将预测的bbox中超出原图的部分裁掉 --> 将分数低于score_threshold的bbox去掉
        :param pred_bbox: yolo输出的bbox信息，shape为(output_size * output_size * gt_per_grid, 5 + num_classes)
        :param test_input_size: 测试尺寸
        :param org_img_shape: 存储格式必须为(h, w)，输入原图的shape
        :return: bboxes
        假设有N个bbox的score大于score_threshold，那么bboxes的shape为(N, 6)，存储格式为(xmin, ymin, xmax, ymax, score, class)
        其中(xmin, ymin, xmax, ymax)的大小都是相对于输入原图的，score = conf * prob，class是bbox所属类别的索引号
        """
        pred_coor = pred_bbox[:, 0:4]
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        # (1)
        # (xmin_org, xmax_org) = ((xmin, xmax) - dw) / resize_ratio
        # (ymin_org, ymax_org) = ((ymin, ymax) - dh) / resize_ratio
        # 需要注意的是，无论我们在训练的时候使用什么数据增强方式，都不影响此处的转换方式
        # 假设我们对输入测试图片使用了转换方式A，那么此处对bbox的转换方式就是方式A的逆向过程
        org_h, org_w = org_img_shape
        resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
        dw = (test_input_size - resize_ratio * org_w) / 2
        dh = (test_input_size - resize_ratio * org_h) / 2
        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        # (2)将预测的bbox中超出原图的部分裁掉
        pred_coor = torch.cat([torch.max(pred_coor[:, :2], torch.zeros((2,)).cuda()),
            torch.min(pred_coor[:, 2:], torch.tensor([org_w - 1, org_h - 1], dtype=torch.float).cuda())], axis=-1)
        # (3)将无效bbox的coor置为0
        invalid_mask = (pred_coor[:, 0] > pred_coor[:, 2]) | (pred_coor[:, 1] > pred_coor[:, 3])

        # (5)将score低于score_threshold的bbox去掉
        classes = torch.argmax(pred_prob, -1)
        scores = pred_conf * pred_prob[torch.arange(pred_coor.shape[0]), classes]
        score_mask = scores > self._score_threshold

        mask = ~invalid_mask & score_mask

        coors = pred_coor[mask]
        scores = scores[mask]
        classes = classes[mask].float()
        # probs = pred_prob[mask]

        bboxes = torch.cat([coors, scores[:, None], classes[:, None]], -1)
        return bboxes
    
    def mAP(self):
        with open(self._test_dataset_file, 'r') as fr:
            imgs_path = [line.strip() for line in fr.readlines() if len(line.strip()) != 0]

        det_results_path = os.path.join('eval', 'results', self._dataset_name)
        if os.path.exists(det_results_path):
            shutil.rmtree(det_results_path)
        os.makedirs(det_results_path)

        total_batch = ceil(len(imgs_path) / self._test_batch_size)
        for batch_index in tqdm(range(total_batch)):
            batch_img_paths = []
            for idx in range(self._test_batch_size):
                img_index = batch_index * self._test_batch_size + idx
                if img_index >= len(imgs_path):
                    break
                batch_img_paths.append(imgs_path[img_index])
            batch_imgs, batch_ori_info = self.__prepare_img(batch_img_paths)
            batch_bboxes_pr = self.__perdict(batch_imgs, batch_ori_info)
            for i, bboxes_pr in enumerate(batch_bboxes_pr):
                for bbox in bboxes_pr:
                    coor = np.array(bbox[:4], dtype=np.int32)
                    score = bbox[4]
                    class_ind = int(bbox[5])
                    class_name = self._classes[class_ind]
                    score = '%.4f' % score
                    xmin, ymin, xmax, ymax = map(str, coor)
                    image_ind = os.path.basename(batch_img_paths[i]).split('.')[0]
                    bbox_mess = ' '.join([image_ind, score, xmin, ymin, xmax, ymax]) + '\n'
                    with open(os.path.join(det_results_path, 'comp3_det_test_' + class_name + '.txt'), 'a') as f:
                        f.write(bbox_mess)
        
        filename = os.path.join('eval', 'results', self._dataset_name, 'comp3_det_test_{:s}.txt')
        cachedir = os.path.join('eval', 'cache')
        annopath = os.path.join(self._dataset_path, 'VOCdevkit', 'VOC2007', 'Annotations', '{:s}.xml')
        imagesetfile = self._test_dataset_file
        APs = {}
        for cls in self._classes:
            _, _, ap = voc_eval.voc_eval(filename, annopath, imagesetfile, cls, cachedir, self._map_iou, False)
            APs[cls] = ap
        if os.path.exists(cachedir):
            shutil.rmtree(cachedir)
        return APs
