from os import path

# yolo
# TRAIN_INPUT_SIZES = [320]
TRAIN_INPUT_SIZES = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
STRIDES = [8, 16, 32]
IOU_LOSS_THRESH = 0.5

# train
BATCH_SIZE = 12
LEARN_RATE_INIT = 1e-4 * BATCH_SIZE / 6
LEARN_RATE_END = 1e-6
WARMUP_EPOCHS = 1
MAX_EPOCHS = int(1.0 * 40 * BATCH_SIZE / 6)

GT_PER_GRID = 3

RESUME = True
RESUME_WEIGHTS = 'weights/VOC_v8/model-31-0.7215.pt'

# test
TEST_INPUT_SIZE = 416
TEST_BATCH_SIZE = 16
SCORE_THRESHOLD = 0.01    # The threshold of the probability of the classes
IOU_THRESHOLD = 0.45     # The threshold of the IOU when implement NMS
MAP_IOU = 0.5

# name and path
DATASET_NAME = 'VOC_v8'
DATASET_PATH = '/home/eleflea/ramdata/Pascal_voc'
TRAIN_DATASET_FILE = '/home/eleflea/ramdata/Pascal_voc/train.txt'
TEST_DATASET_FILE = '/home/eleflea/ramdata/Pascal_voc/2007_test.txt'
WEIGHTS_DIR = path.join('weights', DATASET_NAME)
BACKBONE_WEIGHTS = 'weights/pretrained/mobilenet_v2-b0353104.pth'
# LOG_DIR = 'log'
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

