import argparse

import cv2
import torch

import tools
from config import cfg
from dataset import augment
from eval.evaluator import convert_pred
from model.newyolo import YOLOv3


def add_hook(model, name, hook_fn=None):
    def _warp(filename):
        def save_hook(module, inputs, outputs):
            print(outputs.shape)
            data = outputs.detach().numpy()
            data.tofile('{}.bin'.format(filename))
        return save_hook

    for i, (n, m) in enumerate(model.named_modules()):
        if n == name:
            if hook_fn is None:
                hook_fn = _warp(name)
            print('register for {}'.format(n))
            m.register_forward_hook(hook_fn)

def main(args):
    score_threshold = 0.25
    iou_threshold = 0.45
    class_names = cfg.dataset.classes
    device = torch.device('cpu')

    model = YOLOv3('model/cfg/mobilenetv2-yolo.cfg')
    # print(model)
    state_dict = torch.load(args.weight, map_location=device)
    model.load_state_dict(state_dict['model'])

    image = cv2.imread(args.img)
    # pylint: disable-msg=not-callable
    original_size = torch.tensor(image.shape[:2], device=device, dtype=torch.float32)
    input_size = torch.tensor([args.size, args.size], device=device, dtype=torch.float32)
    preprocess = augment.Compose([
        augment.ToTensor(device),
        augment.Resize([args.size, args.size]),
        # augment.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_image = preprocess(image, [])[0].unsqueeze_(0)
    # input_image = tools.img_preprocess2(image, None, (args.size, args.size), False)
    # input_image = torch.from_numpy(input_image[None, ...]).float()

    model.eval()
    with torch.no_grad():
        batch_pred_bbox = model(input_image)
    batch_pred_bbox = convert_pred(batch_pred_bbox, input_size, original_size)

    batch_bboxes = []
    for pred_bboxes in batch_pred_bbox:
        bboxes = tools.torch_nms(
            pred_bboxes,
            score_threshold,
            iou_threshold,
        ).cpu().numpy()
        batch_bboxes.append(bboxes)

    print(batch_bboxes[0])

    for box in batch_bboxes[0]:
        x1, y1, x2, y2, *_ = [int(n) for n in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = '{} {:.3f}'.format(class_names[int(box[-1])], box[4])
        cv2.putText(image, text, (x1, y1-5), 0, 0.4, (0, 255, 0))
    cv2.imwrite(args.img.rsplit('.', 1)[0] + '_mark.jpg', image)
    # cv2.imshow('show', image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test configuration")
    parser.add_argument('--weight', help='model weight')

    parser.add_argument('--size', help='test image size', type=int, default=512)
    parser.add_argument('--img', help='image path', type=str)
    args = parser.parse_args()
    main(args)
