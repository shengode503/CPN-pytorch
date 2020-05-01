from __future__ import absolute_import, print_function

# import
import os
import math
import time
import torch
import pickle
import numpy as np
from ..data_helper.data_utils import transform_preds


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), 'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]  # e.g. 6
    num_joints = batch_heatmaps.shape[1]  # e.g. 17
    width = batch_heatmaps.shape[3]  # e.g. 64

    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))  # e.g. [6, 17, 64*64]
    idx = np.argmax(heatmaps_reshaped, 2).reshape((batch_size, num_joints, 1))  # Max index.  e.g. [6, 17, 1]
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((batch_size, num_joints, 1))  # Max value.  e.g. [6, 17, 1]

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)  # e.g. [6, 17, 2]
    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))  # max value should > 0
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask  # if max value < 0: coord = 0
    return preds, maxvals


def flip_back(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp
    return output_flipped


# get_final_preds
def get_final_preds(batch_heatmaps, center, scale):

    coords, maxvals = get_max_preds(batch_heatmaps)  # e.g. [6, 17, 2]

    heatmap_height = batch_heatmaps.shape[2]  # e.g. 64
    heatmap_width = batch_heatmaps.shape[3]  # e.g. 64

    # post-processing
    for n in range(coords.shape[0]):  # Batch size
        for p in range(coords.shape[1]):  # pts
            hm = batch_heatmaps[n][p]
            px = int(math.floor(coords[n][p][0] + 0.5))
            py = int(math.floor(coords[n][p][1] + 0.5))
            if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                diff = np.array([hm[py][px+1] - hm[py][px-1], hm[py+1][px]-hm[py-1][px]])
                coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i],
                                   [heatmap_width, heatmap_height])
    return preds, maxvals


def select_device(device='', apex=False):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        cuda_str = 'Using CUDA ' + ('Apex ' if apex else '')  # apex for mixed precision https://github.com/NVIDIA/apex
        for i in range(0, ng):
            if i == 1:
                cuda_str = ' ' * len(cuda_str)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (cuda_str, i, x[i].name, x[i].total_memory / c))
    else:
        print('Using CPU')

    print('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def save_checkpoint(states, prediction, is_best, output_dir, filename='checkpoint', snapshot=None):
    if snapshot and states['epoch'] % snapshot == 0:
        torch.save(states, os.path.join(output_dir, filename + str(states['epoch']) + '.pth'))

    if is_best and 'state_dict' in states:
        torch.save(states, os.path.join(output_dir, 'model_best.pth'))
        # Save Prediction
        with open(os.path.join(output_dir, 'model_best_pred.pkl'), 'wb') as f:
            pickle.dump(prediction, f)




# Logger
class Logger(object):
    def __init__(self, file, states_name):

        # open file
        if os.path.exists(file):
            self.writer = open(file, 'a')
        else:
            self.writer = open(file, 'w')

        # Print State name
        self. _write(states_name)

    def update(self, states):
        self. _write(states)

    def printWrite(self, states, isprint=True):
        localtime = time.asctime(time.localtime(time.time()))
        line = '--------------------------{0}---------------------------'.format(localtime + '\n')
        print(states, file=self.writer)
        self.writer.write('\n')
        self.writer.flush()
        if isprint:
            print(states)

    def close(self):
        self.writer.close()

    def _write(self, states):
        if isinstance(states, list):
            for item in states:
                if isinstance(item, str):
                    self.writer.write(item)
                else:
                    self.writer.write(str(item))
                self.writer.write('\t')
            self.writer.write('\n')
            self.writer.flush()
        else:
            if isinstance(states, str):
                self.writer.write(states)
            else:
                self.writer.write(str(states))
            self.writer.write('\n')
            self.writer.flush()

