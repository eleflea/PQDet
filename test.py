import argparse

import torch

from config import cfg
from eval.evaluator import Evaluator
from model.newyolo import YOLOv3
from dataset.eval_dataset import EvalDataset


def main(config, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLOv3(config.model.cfg_path).to(device)
    state_dict = torch.load(args.weight, map_location=device)
    model.load_state_dict(state_dict['model'])

    eval_dataset = EvalDataset(config)
    evaluator = Evaluator(model, eval_dataset, config)
    model.eval()
    mAP = evaluator.evaluate()
    for kls, acc in mAP.classes.items():
        print('AP@{} = {:.2f}%'.format(kls, acc*100))
    print('mAP = {:.2f}%'.format(mAP.mean*100))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test configuration")
    parser.add_argument('--weight', help='model weight')
    parser.add_argument('--cfg', help='model cfg file')
    parser.add_argument('--size', help='test image size', type=int, default=512)
    parser.add_argument('--iou', help='test AP iou', type=int, default=0.5)
    parser.add_argument('--nms-iou', help='NMS iou', type=int, default=0.45)
    parser.add_argument('--threshold', help='test score threshold', type=float, default=0.01)
    args = parser.parse_args()
    cfg.model.cfg_path = args.cfg
    cfg.eval.input_size = args.size
    cfg.eval.score_threshold = args.threshold
    cfg.eval.iou_threshold = args.nms_iou
    cfg.eval.map_iou = args.iou
    cfg.dataset.eval_txt_file = r'D:\Downloads\VOC\2007_test.txt'
    cfg.freeze()
    main(cfg, args)

#         for bbox in bboxes:
#             coor = np.array(bbox[:4], dtype=np.int32)
#             score = bbox[4]
#             class_ind = int(bbox[5])
#             class_name = self._classes[class_ind]
#             score = '%.4f' % score
#             xmin, ymin, xmax, ymax = map(str, coor)
#             image_ind = file_name.split('.')[0]
#             bbox_mess = ' '.join([image_ind, score, xmin, ymin, xmax, ymax]) + '\n'
#             with open(os.path.join('eval', 'results', 'voc', 'comp3_det_test_' + class_name + '.txt'), 'a') as f:
#                 f.write(bbox_mess)

# filename = os.path.join('eval', 'results', 'voc', 'comp3_det_test_{:s}.txt')
# cachedir = os.path.join('eval', 'cache')
# annopath = 'D:\\Downloads\\VOC\\VOCdevkit\\VOC2007\\Annotations\\{:s}.xml'
# imagesetfile = r'D:\Downloads\VOC\2007_test.txt'
# APs = {}
# for cls in self._classes:
#     _, _, ap = voc_eval.voc_eval(filename, annopath, imagesetfile, cls, cachedir, 0.5, False)
#     APs[cls] = ap
# if os.path.exists(cachedir):
#     shutil.rmtree(cachedir)