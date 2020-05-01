from __future__ import print_function, absolute_import

# Import
import os
import cv2
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt


# Data Clean & Process
def data_cleaner(self, ids, im_ann, anns,
                 is_data_clean=True, is_straight=True):
    if is_data_clean:
        width = im_ann['width']
        height = im_ann['height']

        is_pass = False
        passed_ann = []
        for ann in anns:
            x, y, w, h = ann['bbox'][:4]  # Boundary Box
            cat_check = ann['category_id'] == 1  # Make Sure the Cat is 1:People
            kps_check = max(ann['keypoints']) != 0  # check Key Point (ignore 0 keyPoints)

            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))

            if ann['area'] > 0 and x2 >= x1 and y2 >= y1 and cat_check and kps_check:
                if is_straight:
                    self.data_info.append({'im_ann': im_ann, 'anns': ann})
                else:
                    passed_ann.append(ann)
                is_pass = True

        if is_pass:
            self.pass_idx.append(ids)
            if not is_straight:
                self.data_info.append({'im_ann': im_ann, 'anns': passed_ann})

    else:  # is_clean = False
        if is_straight:
            self.data_info += [{'im_ann': im_ann, 'anns': ann} for ann in anns]
        else:
            self.data_info.append({'im_ann': im_ann, 'anns': anns})


def compute_meanstd(self, im_ann, meanstd_file):
    # if file exist
    if os.path.isfile(meanstd_file):
        meanstd = torch.load(meanstd_file)

    else:  # Compute "Training data's" Mean & Std
        print('''Compute Training Data's Mean & Std !!!''')
        mean = torch.zeros(3)
        std = torch.zeros(3)
        data_len = len(self.data_info)

        for i, ann in enumerate(im_ann):
            index = ann['file_name']

            print('Total_data_num:', data_len, 'Index:', i, 'image_name:', index)

            img_path = os.path.join(self.img_path + index)
            # Load Image
            img = load_image(img_path, is_norm=True, toTesor=True)  # [h, w, c]
            # Compute
            img = img.reshape(img.size(2), -1)
            mean += img.mean(1)
            std += img.std(1)
        mean /= data_len
        std /= data_len

        # Save
        meanstd = {'mean': mean, 'std': std}
        torch.save(meanstd, meanstd_file)
        print('Down!! ', meanstd)

    return meanstd['mean'], meanstd['std']


def process_kp(ann_kps, num_joints):
    # Kps & Kps visible
    kps = np.asarray(ann_kps).reshape([num_joints, 3])
    kp_visible = np.zeros_like(kps)

    vis = kps[:, -1]
    vis[vis > 1] = 1
    kp_visible[:, :2] = np.asarray([vis]*2).T
    kps[:, -1] = 0

    return kps.astype(np.float), kp_visible.astype(np.float)


# Compute Center & Scale
def center_scale(bbox, aspect_ratio, pixel_std):
    # https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/dataset/coco.py#L76
    # aspect_ratio = self.image_width * 1.0 / self.image_height  (w/h)
    # self.image_height = cfg.MODEL.IMAGE_SIZE[1]  <= Input size

    x, y, w, h = bbox
    # Center
    center = np.asarray([x + w * 0.5, y + h * 0.5])  # The Center of Boundary Box
    # Scale
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)

    if center[0] != -1:
        scale *= 1.25

    return center, scale


def half_body_transform(joints, joints_vis, aspect_ratio=1, pixel_std=200,
                        upper_body_ids=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                        lower_body_ids=(11, 12, 13, 14, 15, 16)):

    # joints_vis: 0:empty or 1:exist
    upper_joints = []
    lower_joints = []
    num_joints = len(joints_vis)

    # Upper / Lower Body Part
    for joint_id in range(num_joints):  # 0 ~ 16
        if joints_vis[joint_id][0] > 0:
            if joint_id in upper_body_ids:
                upper_joints.append(joints[joint_id])
            else:
                lower_joints.append(joints[joint_id])

    if np.random.randn() < 0.5 and len(upper_joints) > 2:
        selected_joints = upper_joints
    else:
        selected_joints = lower_joints if len(lower_joints) > 2 else upper_joints

    if len(selected_joints) < 2:
        return None, None

    selected_joints = np.array(selected_joints, dtype=np.float32)  # e.g. shape(6, 3)

    # half body part center is the mean of upper/lower pts
    center = selected_joints.mean(axis=0)[:2]  # e.g. shape(2), [x, y]

    # Calculate the the are of half body part(upper/lower)
    left_top = np.amin(selected_joints, axis=0)  # upper left x/y
    right_bottom = np.amax(selected_joints, axis=0)  # lower right x/y
    w = right_bottom[0] - left_top[0]
    h = right_bottom[1] - left_top[1]

    # aspect_ratio = 1 (self.image_width * 1.0 / self.image_height  (w/h))
    # e.g. if ([w, h] = 60, 36) => ([w, h] = 60, 60)
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    # pixel_std = 200.  e.g. if ([w, h] = 60, 60) => ([w, h] = 60/200, 60/200)
    scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)

    # e.g. if ([w, h] = 60/200, 60/200) => ([w, h] = 60/200 * 1.5, 60/200 * 1.5)
    scale = scale * 1.5
    return center, scale


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip kps
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    j = np.zeros_like(joints)
    j_vis = np.zeros_like(joints_vis)
    for pair in matched_parts:
        j[pair[0], :], j[pair[1], :] = joints[pair[1], :], joints[pair[0], :].copy()
        j_vis[pair[0], :], j_vis[pair[1], :] = joints_vis[pair[1], :], joints_vis[pair[0], :].copy()
    return j*j_vis, j_vis


def get_affine_transform(center, scale, rot, output_size,
                         shift=np.array([0, 0], dtype=np.float32), inv=0):

    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0  # e.g. ([w, h] = 60*1.5/200,  60*1.5/200) => ([w, h] = 60*1.5, 60*1.5)
    src_w = scale_tmp[0]  # e.g. 60*1.5
    dst_w = output_size[0]  # 256
    dst_h = output_size[1]  # 256

    rot_rad = rot * np.pi / 180  # Rotate degree

    # Get Direction
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    # dst
    dst = np.float32([np.asarray([dst_w * 0.5, dst_h * 0.5]),
                      np.asarray([dst_w * 0.5, dst_h * 0.5]) + dst_dir])
    dst = np.vstack((dst, get_3rd_point(dst[0, :], dst[1, :])))

    # src
    src = np.float32([center + scale_tmp * shift,
                      (center + src_dir) + scale_tmp * shift])
    src = np.vstack((src, get_3rd_point(src[0, :], src[1, :])))

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def generate_heatmap(kps, kp_visable, num_joints, in_res,
                     out_res, sigma, target_type='Gaussian'):
    # ------------------------------------------------------------------------------
    # Copyright (c) Microsoft
    # Licensed under the MIT License.
    # Written by Bin Xiao (Bin.Xiao@microsoft.com)
    # ------------------------------------------------------------------------------

    target_weight = np.ones((num_joints, 1), dtype=np.float32)  # [kps_num, 1]
    target_weight[:, 0] = kp_visable[:, 0]  # 0 or 1

    if target_type == 'Gaussian':  # [kps_num x out_w x out_h]
        target = np.zeros((num_joints, out_res[0], out_res[1]), dtype=np.float32)
        tmp_size = sigma * 3
        for joint_id in range(num_joints):  # 0~16
            feat_stride = np.asarray(in_res) / np.asarray(out_res)
            mu_x = int(kps[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(kps[joint_id][1] / feat_stride[1] + 0.5)

            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= out_res[1] or ul[1] >= out_res[0] or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], out_res[1]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], out_res[0]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], out_res[1])
            img_y = max(0, ul[1]), min(br[1], out_res[0])

            if target_weight[joint_id] > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [src_point[0] * cs - src_point[1] * sn,
                  src_point[0] * sn + src_point[1] * cs]
    return src_result


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


# Load Image
def load_image(img_path, is_norm=True, toTesor=False):

    # Load Image [RGB (w*h*c)]
    img = np.asarray(imageio.imread(img_path), dtype=np.float32)

    # norm with / 255
    img = img / 255 if is_norm and img.max() > 1 else img

    # Some Image are only 1 channel (Gray), change to 3 channels
    img = np.asarray([img] * 3).transpose([1, 2, 0]) if np.ndim(img) != 3 else img

    # to Tensor
    img = torch.from_numpy(img) if toTesor else img

    return img


def kp_plot(img, ann):

    fig, ax = plt.subplots(1)
    ax.imshow(img)
    ax.axis('off')

    c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]

    # turn skeleton into zero-based index
    kp = ann[0].reshape([17*3])
    x = kp[0::3]
    y = kp[1::3]
    v = kp[2::3]

    # KeyPoint
    ax.plot(x[v > 0], y[v > 0], 'o', markersize=8, markerfacecolor=c, markeredgecolor='k', markeredgewidth=2)
    ax.plot(x[v > 1], y[v > 1], 'o', markersize=8, markerfacecolor=c, markeredgecolor=c, markeredgewidth=2)
