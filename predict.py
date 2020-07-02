import argparse

import cv2
import torch

import tools
from config import cfg
from dataset import EVAL_AUGMENT_REGISTER, RECOVER_BBOXES_REGISTER
from model.interpreter import DetectionModel


def main(args):
    dataset_name = args.dataset
    score_threshold = args.threshold
    iou_threshold = args.nms_iou
    class_names = cfg.dataset.classes
    device = torch.device('cpu')

    model = DetectionModel(args.cfg)
    # print(model)
    state_dict = torch.load(args.weight, map_location=device)
    torch.nn.DataParallel(model).load_state_dict(state_dict['model'])
    model.to(device)

    image = cv2.imread(args.img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # pylint: disable-msg=not-callable
    original_size = torch.tensor(image.shape[:2], device=device, dtype=torch.float32)
    input_size = torch.tensor([args.size, args.size], device=device, dtype=torch.float32)
    preprocess = EVAL_AUGMENT_REGISTER[dataset_name](args.size, device)
    input_image = preprocess(image, [])[0].unsqueeze_(0)

    model.eval()
    with torch.no_grad():
        batch_pred_bbox = model(input_image)
    batch_pred_bbox = RECOVER_BBOXES_REGISTER[dataset_name](batch_pred_bbox, input_size, original_size)

    batch_bboxes = []
    for pred_bboxes in batch_pred_bbox:
        bboxes = tools.torch_nms(
            pred_bboxes,
            score_threshold,
            iou_threshold,
        ).cpu().numpy()
        batch_bboxes.append(bboxes)

    print(f'detect {len(batch_bboxes[0])} objects.')
    print(batch_bboxes[0])

    for box in batch_bboxes[0]:
        x1, y1, x2, y2, *_ = [int(n) for n in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = '{} {:.3f}'.format(class_names[int(box[-1])], box[4])
        cv2.putText(image, text, (x1, y1-5), 0, 0.4, (0, 255, 0))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.img.rsplit('.', 1)[0] + '_mark.jpg', image)
    # cv2.imshow('show', image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test configuration")
    parser.add_argument('--yaml', default='yamls/voc.yaml', required=False)
    parser.add_argument('--dataset', default='voc', help='dataset name')
    parser.add_argument('--cfg', help='model cfg file')
    parser.add_argument('--weight', help='model weight')

    parser.add_argument('--size', help='test image size', type=int, default=512)
    parser.add_argument('--nms-iou', help='NMS iou', type=int, default=0.45)
    parser.add_argument('--threshold', help='predict score threshold', type=float, default=0.25)
    parser.add_argument('--img', help='image path', type=str)
    args = parser.parse_args()
    if args.yaml:
        cfg.merge_from_file(args.yaml)
    cfg.freeze()
    main(args)
