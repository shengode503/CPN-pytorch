# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .utils import get_max_preds


def calc_dists(preds, target, normalize):
    # e.g. preds, target = [6, 17, 2], normalize = [6, 2]
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))  # e.g. [17, 6]

    for n in range(preds.shape[0]):  # Batch size.  e.g. 6
        for c in range(preds.shape[1]):  # kps_num.  e.g. 17
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:  # if kps exist & > 1.
                # normalization
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                # Calcu Dist
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    # Output.shape: [batch_size, 17, 64, 64]
    # target.shape: [batch_size, 17, 64, 64]
    idx = list(range(output.shape[1]))  # pks_idx [0~16]
    norm = 1.0
    if hm_type == 'gaussian':  # HeatMap type
        # get the coord of max value in each part's HeatMap.  (HeatMap -> coord)  e.g.(6, 17, 2)
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        # Normalization
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10  # e.g.(6, 2)  w/10, h/10

    # Calculate the distance between predictions and targets.  e.g.(17, 6)
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))  # e.g. 18
    avg_acc = 0
    cnt = 0
    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]], thr)
        if acc[i + 1] >= 0:
            avg_acc += acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc

    return acc, avg_acc, cnt, pred


