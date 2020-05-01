# Import
from lib.models.CPN.cpn import get_pose_net
from lib.data_helper.data_utils import *
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from lib.utils.utils import *
from lib.config import cfg
import cv2


# Load Data
img_path = 'dataset/test_data/4.jpg'
img = load_image(img_path, is_norm=True, toTesor=False)

plt.figure(0)
plt.imshow(img)

# Person Detector
# Using person detector to get the bbox of person.
# -----------------------------------------------------
bbox = []


# ======================================================================

' Some Parameters '
in_res = cfg.MODEL.IMAGE_SIZE  # input resolution.
in_width, in_high = in_res[1], in_res[0]
aspect_ratio = in_width * 1.0 / in_high

meanstd_path = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.MEAN_STD_PATH)

' Data Process '
# Center & Scale
pixel_std = 200
x, y, w, h = bbox
c = np.asarray([x + w * 0.5, y + h * 0.5])[None]

if w > aspect_ratio * h:
    h = w * 1.0 / aspect_ratio
elif w < aspect_ratio * h:
    w = h * aspect_ratio
s = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)[None]

# Image affine
trans = get_affine_transform(c, s, 0, [in_width, in_high])  # get_affine_transform (M)
input_img = cv2.warpAffine(img, trans, (int(in_width), int(in_high)), flags=cv2.INTER_LINEAR)

# pre-process
meanstd = torch.load(meanstd_path)
normalize = transforms.Normalize(mean=meanstd['mean'].numpy().tolist(),
                                 std=meanstd['std'].numpy().tolist())
trans = transforms.Compose([transforms.ToTensor(), normalize])
input_img = trans(input_img)

' Inference '
# Select Device
device = select_device(str(cfg.DEVICE))  # str(cfg.DEVICE)
cudnn.benchmark = cfg.CUDNN.BENCHMARK

# Model
model = get_pose_net(cfg, is_train=False)
state = torch.load(os.path.join(cfg.CHECKPOINT_PATH, 'model_best.pth'), map_location=device)
model.load_state_dict(state['state_dict'])

model = model.to(device)
model.eval()

# Inference
with torch.no_grad():

    # Select device
    input_img = input_img.to(device)

    # predict
    _, outputs = model(input_img[None])
    output = outputs[-1].cpu()

    preds, maxvals = get_final_preds(output.clone().cpu().numpy(), c, s)
    res = np.hstack((preds[0], maxvals[0]))  # for batch = 1.

    # Plot
    plt.figure(1)
    plt.imshow(img)

    thre = 0.1
    for r in res:
        if r[2] > thre:
            plt.plot(int(r[0]), int(r[1]), 'o', markersize=8, markeredgecolor='k', markeredgewidth=2)

    line_pairs = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                  [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                  [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]

    for line in line_pairs:
        kp1 = res[line[0]]
        kp2 = res[line[1]]
        if kp1[2] > thre and kp2[2] > thre:
            plt.plot([kp1[0], kp2[0]], [kp1[1], kp2[1]], 'r', markersize=4, markeredgecolor='k', markeredgewidth=2)
