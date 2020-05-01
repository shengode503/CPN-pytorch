# http://cocodataset.org/#detection-eval
# https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb

from __future__ import print_function, absolute_import


# Import
import torch.backends.cudnn as cudnn
from lib.data_helper.COCO_loader import Coco
from lib.models.Loss.mseLoss import JointsMSELoss
from lib.models.CPN.cpn import get_pose_net
from pycocotools.cocoeval import COCOeval
from lib.train.trainer import validate
from collections import OrderedDict
from pycocotools.coco import COCO
from lib.utils.utils import *
from lib.config import cfg


# Select Device
cudnn.benchmark = cfg.CUDNN.BENCHMARK
device = select_device(str(cfg.DEVICE))  # 'cpu', '0', '1'  str(cfg.DEVICE)

# n joints
njoints = cfg.DATASET.NJOINTS

# Model (Stacks Hourglass Network)
model = get_pose_net(cfg, is_train=False)
state = torch.load(os.path.join(cfg.CHECKPOINT_PATH, 'model_best.pth'), map_location=device)
model.load_state_dict(state['state_dict'])
model = model.to(device)

print('--> Load Best Model ||  Epoch: {0},  Model: {1},  Accuracy: {2}'.format(
    state['epoch'], state['model'], state['perf']))

# loss function (criterion) and optimizer
criterion = JointsMSELoss(use_target_weight=True).to(device)

# Data Loader
val_loader = torch.utils.data.DataLoader(
    Coco(cfg, is_train=False, transform=None,
         is_clean=False,
         is_straight=True),
    batch_size=cfg.TRAIN.TRAIN_BATCH, shuffle=False,
    num_workers=cfg.WORKERS, pin_memory=cfg.PIN_MEMORY)

# Validation
valid_loss, valid_acc, prediction = validate(cfg, val_loader, model, criterion, device,
                                             print_freq=999999,
                                             flip=cfg.TEST.FLIP_TEST,
                                             shift_hpval=cfg.TEST.SHIFT_RES)

# -------------------------------------------------

# # OKS
# import pickle
# with open('prediction.pkl', 'wb') as f:
#     pickle.dump(prediction, f)

# import pickle
# with open('prediction.pkl', 'rb') as f:
#     prediction = pickle.load(f)

# -------------------------------------------------

results = []
coco = COCO(os.path.join(cfg.DATASET.ROOT, cfg.DATASET.TESTSET))
anns = coco.loadAnns(prediction['imgIdList'])
image_id = [ann['image_id'] for ann in anns]

for idx in range(len(image_id)):

    kps = prediction['all_preds'][idx, :, :].flatten().tolist()

    item = {"image_id": int(image_id[idx]),
            "category_id": int(1),
            "keypoints": kps,
            "score": 1.0}
    results.append(item)

# -------------------------------------------------

in_vis_thre = 0.2
for i in range(len(results)):

    img_kpts = results[i]['keypoints']
    kpt = np.array(img_kpts).reshape(17, 3)
    box_score = results[i]['score']

    kpt_score = 0
    valid_num = 0
    # each joint for bbox
    for n_jt in range(cfg.DATASET.NJOINTS):

        # score
        t_s = kpt[n_jt][2]
        if t_s > in_vis_thre:
            kpt_score += t_s
            valid_num += 1

    if valid_num != 0:
        kpt_score = kpt_score / valid_num

    results[i]['score'] = kpt_score * box_score

# -----------------------------------------------------------------------

cocoGt = COCO(os.path.join(cfg.DATASET.ROOT, cfg.DATASET.TESTSET))
cocoDt = cocoGt.loadRes(results)

cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
cocoEval.params.useSegm = None

cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()


# # Plot format
# # -----------------------------------------------------------------------
#
# # markdown format output
# def _print_name_value(name_value, full_arch_name):
#     names = name_value.keys()
#     values = name_value.values()
#     num_values = len(name_value)
#     print(
#         '| Arch ' +
#         ' '.join(['| {}'.format(name) for name in names]) +
#         ' |'
#     )
#     print('|---' * (num_values+1) + '|')
#
#     if len(full_arch_name) > 15:
#         full_arch_name = full_arch_name[:8] + '...'
#     print(
#         '| ' + full_arch_name + ' ' +
#         ' '.join(['| {:.3f}'.format(value) for value in values]) +
#          ' |'
#     )
#
# stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
#
# info_str = []
# for ind, name in enumerate(stats_names):
#     info_str.append((name, cocoEval.stats[ind]))
#
# name_values = OrderedDict(info_str)
# model_name = 'Hourglass'
#
# if isinstance(name_values, list):
#     for name_value in name_values:
#         _print_name_value(name_value, model_name)
# else:
#     _print_name_value(name_values, model_name)







































# ## Test
#
# ann_gt = cocoGt.anns
# ann_dt = cocoDt.anns
#
# GG = []
# FK = []
# for idx, key in enumerate(list(ann_gt.keys())):
#     ann = ann_gt[key]
#
#     if ann['image_id'] == 1000:
#         GG.append(ann)
#
#     agg = ann_dt[idx + 1]
#     if agg['image_id'] == 1000:
#         FK.append(agg)
#
# # plot
# import cv2
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
#
# file = 'C:\\Users\\ikk\\Desktop\\CV_Paper\\Human Pose Estimation\\Code\\Human_Pose_Estimation-pytorch_v4\\dataset\\mscoco\\val2017\\'
# img = cv2.cvtColor(cv2.imread(file + '000000001000.jpg'), cv2.COLOR_BGR2RGB)
#
# # bbox gt
# plt.figure(0)
# plt.imshow(img)
#
# for j, g in enumerate(GG):
#     x, y, w, h = g['bbox']
#
#     currentAxis = plt.gca()
#     rect = patches.Rectangle((int(x), int(y)), w, h, linewidth=1, edgecolor='r', facecolor='none')
#     currentAxis.add_patch(rect)
#
#     kps = np.asarray(g['keypoints']).reshape(17, 3)
#     for i in range(17):
#         ggg = kps[i]
#         plt.plot(ggg[0], ggg[1], 'b*')
#
#     if j == 2:
#         break
#
#
#
# # bbox dt
# plt.figure(1)
# plt.imshow(img)
#
# for j, fk in enumerate(FK):
#     x, y, w, h = fk['bbox']
#
#     currentAxis = plt.gca()
#     rect = patches.Rectangle((int(x), int(y)), w, h, linewidth=1, edgecolor='r', facecolor='none')
#     currentAxis.add_patch(rect)
#
#     kps = np.asarray(fk['keypoints']).reshape(17, 3)
#     for i in range(17):
#         ggg = kps[i]
#         if ggg[2] > 0.3:
#             plt.plot(ggg[0], ggg[1], 'b*')
#
#     if j == 2:
#         break
