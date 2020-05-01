# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

import torch.nn as nn
import torch

class JointsMSELoss(nn.Module):
    # With L2-Dist
    def __init__(self, use_target_weight=True):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        # Output.shape = [batch, Joints, feature_map_w, feature_map_h]
        # target.shape = [batch, Joints, feature_map_w, feature_map_h]

        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(heatmap_pred.mul(target_weight[:, idx][:, None]),
                                             heatmap_gt.mul(target_weight[:, idx][:, None]))
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
        return loss / num_joints


class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight=True, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx][:, None]),
                    heatmap_gt.mul(target_weight[:, idx][:, None])))
            else:
                loss.append(0.5 * self.criterion(heatmap_pred, heatmap_gt))

        loss = [(l[None] if l.ndim == 1 else l).mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)
        return self.ohkm(loss)



if __name__ == '__main__':
    import torch

    criterion = JointsMSELoss().to('cpu')
    criterion_ohkm = JointsOHKMMSELoss(use_target_weight=True).to('cpu')

    pred_heatmap = torch.randn([6, 16, 64, 64])
    gt_heatmap = 10 * torch.randn([6, 16, 64, 64])

    loss_w = torch.FloatTensor([1]*16)[None]

    loss = criterion(pred_heatmap, gt_heatmap, loss_w)
    loss_ohkm = criterion_ohkm(pred_heatmap, gt_heatmap, loss_w)
