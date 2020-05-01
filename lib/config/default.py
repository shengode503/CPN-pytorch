# Import
import os
from yacs.config import CfgNode as CN

_C = CN()

# Device
_C.DEVICE = 0  # 0, 1, 'cpu'
_C.WORKERS = 0
_C.PIN_MEMORY = True

_C.PRINT_FREQ = 30

# dir path
_C.CHECKPOINT_PATH = 'result/checkpoints/'
_C.CHECK_RES_PATH = 'result/check_result/'

' CUDNN '
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

' DATASET '
_C.DATASET = CN()
_C.DATASET.NAME = 'coco'  # coco / mpii
_C.DATASET.NJOINTS = 17 if _C.DATASET.NAME == 'coco' else 16
_C.DATASET.IS_TRAIN = True
_C.DATASET.DATA_FORMAT = 'jpg'

# Path
_C.DATASET.ROOT = 'dataset/mscoco/'
_C.DATASET.TRAINSET = 'annotations/person_keypoints_train2017.json'  # person_keypoints_train2017.json
_C.DATASET.TESTSET = 'annotations/person_keypoints_val2017.json'
_C.DATASET.IMAGE = ['train2017/',  # Train path # train2017/
                    'val2017/'  # Val path
                    ]
_C.DATASET.MEAN_STD_PATH = 'mean.pth.tar'

# training data augmentation
_C.DATASET.IS_FLIP = True
_C.DATASET.FLIP_PROB = 0
_C.DATASET.SCALE_FACTOR = 0.30
_C.DATASET.ROT_FACTOR = 45

_C.DATASET.PROB_HALF_BODY = 0.0
_C.DATASET.NUM_JOINTS_HALF_BODY = 8
_C.DATASET.COLOR_RGB = False

_C.DATASET.DROP_OOB = False  # Drop out of boundary keypoints.

' Model '
_C.MODEL = CN()
_C.MODEL.NAME = 'cpn'  # hourglass, cpn, simple_baseline, hrnet
_C.MODEL.IMAGE_SIZE = [384, 288]  # height * width, ex: 256 * 192
_C.MODEL.HEATMAP_SIZE = [96, 72]  # height * width, ex: 64 * 43

# Generate HeatMap
_C.MODEL.SIGMA = 1
_C.MODEL.TARGET_TYPE = 'Gaussian'
_C.MODEL.SIGMA_DECAY = 0

_C.MODEL.PRETRAIN_RES_ROOT = 'pretrain_model/pretrain_resnet/'

_C.MODEL.EXTRA = CN(new_allowed=True)
from .models import MODEL_EXTRAS
_C.MODEL.EXTRA = MODEL_EXTRAS[_C.MODEL.NAME]

' Logger '
_C.LOGGER = CN()
_C.LOGGER.EPOCH_LOG_POTH = 'result/logs/epoch_logger.txt'
_C.LOGGER.TRAIN_LOGER = 'result/logs/train_logger.txt'
_C.LOGGER.VAL_LOGGER = 'result/logs/val_logger.txt'
_C.LOGGER.TRAIN_EVENT_LOG_POTH = 'result/logs/train_event_logger.txt'

' TRAIN '
_C.TRAIN = CN()

_C.TRAIN.FOR_PARALLEL = False

_C.TRAIN.DATA_SHUFFLE = True

_C.TRAIN.OHKM = True

# Learning Rate
_C.TRAIN.LR = 5e-4
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.SCHEDULE = [60, 90]

# Optimizer
_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0
_C.TRAIN.WEIGHT_DECAY = 0


# Epoch
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140

# Batch
_C.TRAIN.TRAIN_BATCH = 16  # !!!

# save
_C.TRAIN.snapshot = 5  # save check point's freq.
_C.TRAIN.CHECK_RES_FREQ = 3  # for every 3 epoch.
_C.TRAIN.CHECK_RES_PATH = 'result/check_result/'


_C.TRAIN.IS_DATA_CLEAN = True
_C.TRAIN.IS_STRAIGHT = False

' Testing '
_C.TEST = CN()
_C.TEST.FLIP_TEST = True
_C.TEST.SHIFT_RES = 1  # 'The Shifting value of HeatMap'


def get_cfg_defaults():
  return _C.clone()


if __name__ == '__main__':

    # # Save
    cfg = get_cfg_defaults()
    cfg.freeze()

    root = ''
    with open(os.path.join(root, 'experiment/experiment_0.yaml'), 'w') as f:
        print(cfg, file=f)

    # # Load
    # cfg = get_cfg_defaults()
    # root = ''
    # cfg.merge_from_file(os.path.join(root, 'experiment/experiment_0.yaml'))
    # cfg.freeze()






