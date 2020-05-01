from __future__ import print_function, absolute_import

# Import
import random
from .data_utils import *
from pycocotools.coco import COCO
import torchvision.transforms as transforms


# Ms-coco
class Coco(torch.utils.data.Dataset):
    '''
    # The Ms-coco DataLoader #

    "keypoints": { 0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
                   5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
                   9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
                   13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"
                  },
    "skeleton": [[16,14], [14,12], [17,15], [15,13], [12,13],
                [6,12], [7,13], [6,7], [6,8], [7,9], [8,10],
                [9,11], [2,3], [1,2], [1,3], [2,4], [3,5], [4,6], [5,7]]
    '''
    def __init__(self, cfg,
                 is_clean=True, is_straight=True,
                 is_train=True, transform=None):

        data_cfg = cfg.DATASET

        # Number of Joints.
        self.num_joints = 17

        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
        self.upper_idx = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.lower_idx = (11, 12, 13, 14, 15, 16)

        # training set or test set
        self.is_train = is_train

        # Some data process (use torch.transform)
        self.transform = transform

        # with data clean?
        self.is_clean = is_clean

        # straight the data?
        self.is_straight = is_straight

        ' Image & anno Path '
        self.mean_std_path = data_cfg.MEAN_STD_PATH  # train data's mean & std.
        self.img_path = os.path.join(data_cfg.ROOT, data_cfg.IMAGE[0 if is_train else 1])
        annPath = os.path.join(data_cfg.ROOT, data_cfg.TRAINSET if is_train else data_cfg.TESTSET)

        # Input & Output shape
        self.in_res = cfg.MODEL.IMAGE_SIZE  # input resolution.
        self.out_res = cfg.MODEL.HEATMAP_SIZE  # output resolution.
        self.in_width, self.in_high = self.in_res[1], self.in_res[0]

        ' Augmentation Factors '
        self.rot_factor = data_cfg.ROT_FACTOR
        self.scale_factor = data_cfg.SCALE_FACTOR
        self.aspect_ratio = self.in_width * 1.0 / self.in_high

        self.is_flip = data_cfg.IS_FLIP
        self.flip_prob = data_cfg.FLIP_PROB

        self.num_joints_half_body = data_cfg.NUM_JOINTS_HALF_BODY
        self.prob_half_body = data_cfg.PROB_HALF_BODY

        # drop out the kp is out of boundary
        self.dropout_oob_kp = data_cfg.DROP_OOB

        # Generate HeatMap
        self.sigma = cfg.MODEL.SIGMA
        self.target_type = cfg.MODEL.TARGET_TYPE

        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # load coco api
        coco = COCO(annPath)

        cat_ids = coco.getCatIds(catNms=['person'])  # [1]
        img_ids = coco.getImgIds(catIds=cat_ids)

        self.pass_idx = []
        self.data_info = []
        for ids in img_ids:
            im_ann = coco.loadImgs(ids)[0]
            anns = coco.loadAnns(coco.getAnnIds(imgIds=ids, catIds=cat_ids))

            # data cleaner
            data_cleaner(self, ids, im_ann, anns,
                         self.is_clean,
                         self.is_straight)

            del im_ann, anns

        # Load Data Mean & Std
        print('-->', os.path.join(data_cfg.ROOT, self.mean_std_path))
        self.mean, self.std = compute_meanstd(self, coco.loadImgs(self.pass_idx),
                                              os.path.join(data_cfg.ROOT, self.mean_std_path))

        # del
        del coco, annPath, cat_ids, img_ids, self.pass_idx,

    def __getitem__(self, index):

        info = self.data_info[index]
        im_ann = info['im_ann']
        ann = info['anns']

        # random pick one person's info.
        if not self.is_straight:
            rand_idx = np.random.randint(0, len(ann), 1)[0] if len(ann) > 1 else 0
            ann = ann[rand_idx]

        # --------------------------------------------------------

        # c: bbox_center, s: bbox_scale
        c, s = center_scale(ann['bbox'], self.aspect_ratio, pixel_std=200)

        # Process kp
        kps, kps_visible = process_kp(ann['keypoints'], self.num_joints)

        # Load Image
        img = load_image(os.path.join(self.img_path, im_ann['file_name']), is_norm=True, toTesor=False)

        # ---------------------------------------------------------------------------------------------

        # Data Augmentation
        r = 0
        if self.is_train:
            ''' Half Body Transform '''
            # if kps_num > NUM_JOINTS_HALF_BODY (threshold=8) & probability of processing (threshold=0)
            # calculate the center & scale of part of half body. and replace the bbox's c/s.
            if np.sum(kps_visible[:, 0]) > self.num_joints_half_body and np.random.rand() < self.prob_half_body:
                c_half_body, s_half_body = half_body_transform(kps, kps_visible)

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body
            c, s = np.float32(c), np.float32(s)

            # Flip
            if self.is_flip and random.random() <= self.flip_prob:
                # Flip Image
                img = img[:, ::-1, :]
                # Flip Kps
                kps, kps_visible = fliplr_joints(kps, kps_visible, img.shape[1], self.flip_pairs)
                c[0] = img.shape[1] - c[0] - 1

            # Scaling & Rotate
            sf = self.scale_factor  # SCALE_FACTOR = 0.25
            rf = self.rot_factor  # ROT_FACTOR = 30
            s *= np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)  # a random value at 1-0.25 ~ 1+0.25
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0  # -60 ~ 60

        # Affine Transform
        trans = get_affine_transform(c, s, r, [self.in_width, self.in_high])  # get_affine_transform (M)
        input_img = cv2.warpAffine(img, trans, (int(self.in_width), int(self.in_high)), flags=cv2.INTER_LINEAR)

        for idx in range(self.num_joints):
            if kps_visible[idx, 0] > 0.0:
                kps[idx, 0:2] = affine_transform(kps[idx, 0:2], trans)

        # Normalize
        normalize = transforms.Normalize(mean=self.mean.numpy().tolist(),
                                         std=self.std.numpy().tolist())
        trans = transforms.Compose([transforms.ToTensor(), normalize])
        input_img = trans(input_img)

        # Some data process (use torch.transform)
        input_img = self.transform(input_img) if self.transform else input_img

        # Some kps will out of img's boundary
        if self.dropout_oob_kp:
            for idx in range(len(kps)):
                if kps[idx, 0] > self.in_width - 1 or kps[idx, 1] > self.in_high - 1 \
                        or kps[idx, 0] < 0 or kps[idx, 1] < 0:
                    kps[idx, :] = np.zeros([3])
                    kps_visible[idx, :] = np.zeros([3])

        # Generate heatmap
        target, target_weight = generate_heatmap(kps, kps_visible, self.num_joints,
                                                 self.in_res, self.out_res, self.sigma)

        # GaussianBlur (For CPN)
        gk9 = cv2.GaussianBlur(target, (13, 13), 0)
        gk11 = cv2.GaussianBlur(target, (17, 17), 0)
        gk15 = cv2.GaussianBlur(target, (23, 23), 0)
        target = [torch.from_numpy(gk15), torch.from_numpy(gk11),
                  torch.from_numpy(gk9), torch.from_numpy(target)]

        # Information
        retain = {'id': ann['id'],
                  'image_name': im_ann['file_name'],
                  'bbox': ann['bbox'],
                  'joints': kps,
                  'joints_vis': kps_visible,
                  'center': c,
                  'scale': s,
                  'rotation': r}
        return input_img, target, target_weight, retain

    def __len__(self):
        return len(self.data_info)





if __name__ == '__main__':

    import cv2
    from lib.config import cfg
    import matplotlib.pyplot as plt


    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        Coco(cfg, is_train=True),
        batch_size=10, shuffle=False,
        num_workers=0, pin_memory=False)

    for i, (input_img, target, target_weight, info) in enumerate(train_loader):
        if i == 2:
            break

    idx = 1

    img = input_img[idx].permute(1, 2, 0).numpy()
    tar = np.sum(target[idx].numpy(), 0)

    import matplotlib.pyplot as plt
    plt.figure(0)
    plt.imshow(img)

    plt.figure(1)
    plt.imshow(tar)
