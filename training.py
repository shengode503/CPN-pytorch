from __future__ import print_function, absolute_import

# Import
import torch.backends.cudnn as cudnn
from lib.data_helper.COCO_loader import Coco
from lib.models.Loss.mseLoss import JointsMSELoss, JointsOHKMMSELoss
from lib.models.CPN.cpn import get_pose_net
from lib.train.trainer import train, validate
from lib.utils.utils import *
from lib.config import cfg
import pickle


# Logger
states_name = ['epoch', 'lr', 'train_loss', 'train_acc',
               'valid_loss', 'valid_acc', 'train_time', 'val_time']
epoch_logger = Logger(cfg.LOGGER.EPOCH_LOG_POTH, states_name)
train_event_logger = Logger(cfg.LOGGER.TRAIN_EVENT_LOG_POTH, '=====-- Train Event Logger --=====')

# Select Device
cudnn.benchmark = cfg.CUDNN.BENCHMARK
device = select_device(str(cfg.DEVICE))  # 'cpu', '0', '1'
train_event_logger.printWrite('--> Select Device:  {0}'.format(cfg.DEVICE))  # Logger

# create checkpoint dir
if not os.path.isdir(cfg.CHECKPOINT_PATH):
    os.makedirs(cfg.CHECKPOINT_PATH)
if not os.path.isdir(cfg.CHECK_RES_PATH):
    os.makedirs(cfg.CHECK_RES_PATH)

# n joints
njoints = cfg.DATASET.NJOINTS
train_event_logger.printWrite('--> Dataset:  {0},  njoints:  {1}'.format(cfg.DATASET.NAME, njoints))  # Logger

# Model
model = get_pose_net(cfg, is_train=True)
train_event_logger.printWrite('--> Model:  {0}.'.format(cfg.MODEL.NAME))  # Logger

# Parallel (GPU) Training
model = torch.nn.DataParallel(model).to(device) if cfg.TRAIN.FOR_PARALLEL else model.to(device)
train_event_logger.printWrite('--> Model with Parallel? {0}.'.format(cfg.TRAIN.FOR_PARALLEL))  # Logger

# loss function (criterion) and optimizer
criterion = JointsMSELoss(use_target_weight=True).to(device)
criterion_okhm = JointsOHKMMSELoss(use_target_weight=True).to(device)

# Choose the Optimizer
if cfg.TRAIN.OPTIMIZER == 'rms':
    optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.TRAIN.LR,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)
elif cfg.TRAIN.OPTIMIZER == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)

elif cfg.TRAIN.OPTIMIZER == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR)

else:
    print('Unknown solver: {}'.format(cfg.TRAIN.OPTIMIZER))
    assert False
train_event_logger.printWrite('--> Optimizer: {0}.'.format(cfg.TRAIN.OPTIMIZER))  # Logger

# DataLoader
train_event_logger.printWrite('--> Load train_loader', isprint=False)  # Logger
train_loader = torch.utils.data.DataLoader(
    Coco(cfg, is_train=True, transform=None,
         is_clean=cfg.TRAIN.IS_DATA_CLEAN,
         is_straight=cfg.TRAIN.IS_STRAIGHT),
    batch_size=cfg.TRAIN.TRAIN_BATCH, shuffle=True,
    num_workers=cfg.WORKERS, pin_memory=cfg.PIN_MEMORY)

train_event_logger.printWrite('--> Load val_loader', isprint=False)  # Logger
val_loader = torch.utils.data.DataLoader(
    Coco(cfg, is_train=False, transform=None,
         is_clean=False,
         is_straight=True),
    batch_size=cfg.TRAIN.TRAIN_BATCH, shuffle=False,
    num_workers=cfg.WORKERS, pin_memory=cfg.PIN_MEMORY)

# -------------------------------------------------------------------------------------------

# Train & Val
idx = []
best_acc = 0
lr = cfg.TRAIN.LR
for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):

    # Adjust Learning Rate.
    lr = adjust_learning_rate(optimizer, epoch, lr, cfg.TRAIN.SCHEDULE, cfg.TRAIN.LR_FACTOR)
    print('\nEpoch: %d | Learn Rate: %.8f' % (epoch + 1, lr))

    # Train on 1 epoch
    train_event_logger.printWrite('--> Training!! Epoch:{0}.'.format(epoch), isprint=False)  # Logger
    end = time.time()
    train_loss, train_acc = train(cfg, train_loader, model, criterion, criterion_okhm, optimizer,
                                  epoch, device, print_freq=cfg.PRINT_FREQ)
    train_time = time.time() - end
    end = time.time()
    train_event_logger.printWrite('# Training!! Epoch:{0}. (Done!)'.format(epoch), isprint=False)  # Logger

    # Evaluate
    train_event_logger.printWrite('--> Validate!! Epoch:{0}.'.format(epoch), isprint=False)  # Logger
    valid_loss, valid_acc, prediction = validate(cfg, val_loader, model, criterion, device,
                                                 print_freq=cfg.PRINT_FREQ,
                                                 flip=cfg.TEST.FLIP_TEST,
                                                 shift_hpval=cfg.TEST.SHIFT_RES)
    val_time = time.time() - end
    train_event_logger.printWrite('# Validate!! Epoch:{0}. (Done!)'.format(epoch), isprint=False)  # Logger

    # Logger
    epoch_logger.update([epoch, lr, train_loss, train_acc, valid_loss, valid_acc, train_time, val_time])

    # remember best acc and save checkpoint
    is_best = valid_acc > best_acc
    best_acc = max(valid_acc, best_acc)

    states = {'epoch': epoch + 1,
              'model': cfg.MODEL.NAME,
              'state_dict': model.state_dict(),
              'perf': best_acc,
              'optimizer': optimizer.state_dict()}
    save_checkpoint(states, prediction, is_best, snapshot=cfg.TRAIN.snapshot, output_dir=cfg.CHECKPOINT_PATH)

    # Test: Check the val result when training.
    if cfg.TRAIN.CHECK_RES_FREQ != 0:
        if epoch % cfg.TRAIN.CHECK_RES_FREQ == 0:
            # Random Pick 20 samples.
            imgNameList = prediction['imgNameList']
            imgIdList = prediction['imgIdList']
            rand_idx = np.random.randint(0, len(imgNameList), 20)
            samples = {'imgId': [imgIdList[i] for i in rand_idx],
                       'imgName': [imgNameList[i] for i in rand_idx],
                       'all_preds': prediction['all_preds'][rand_idx],
                       'all_boxes': prediction['all_boxes'][rand_idx]}

            with open(os.path.join(cfg.TRAIN.CHECK_RES_PATH, 'ck_result_epoch_{0}'.format(epoch)), 'wb') as f:
                pickle.dump(prediction, f)

# Close
epoch_logger.close()
train_event_logger.close()
