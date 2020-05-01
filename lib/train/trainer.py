from __future__ import absolute_import

# import
import time
import torch
import numpy as np
from ..utils.utils import AverageMeter, flip_back, get_final_preds, Logger
from ..utils.progress.progress.bar import Bar  # https://github.com/verigak/progress
from ..utils.evaluate import accuracy


# trainer
def train(cfg, train_loader, model, criterion, criterion_okhm, optimizer, epoch, device, print_freq=1):
    # Batch data container.
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    # Logger
    train_logger = Logger(cfg.LOGGER.TRAIN_LOGER, '=====-- Train Logger {0} --====='.format(epoch))

    # switch to train mode.
    model.train()

    # visualize tool
    bar = Bar('Train', max=len(train_loader))

    end = time.time()
    for i, (input_img, target, target_weight, info) in enumerate(train_loader):

        batch_size = len(input_img)

        # measure data loading time
        data_time.update(time.time() - end)

        # choose device
        input_img = input_img.to(device)
        target = [tar.to(device, non_blocking=True) for tar in target]
        target_weight = target_weight.to(device, non_blocking=True)

        # prediction
        global_outputs, refine_output = model(input_img)

        ' Loss '
        loss = 0
        # GlobalNet Loss
        for out, tar in zip(global_outputs, target):
            loss += criterion(out, tar, target_weight)

        # RefineNet Loss
        loss += criterion_okhm(refine_output, target[-1], target_weight)
        losses.update(loss.item(), batch_size)  # record loss

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy
        _, avg_acc, cnt, pred = accuracy(refine_output.detach().cpu().numpy(),
                                         target[-1].detach().cpu().numpy())
        acces.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        msg = 'Epoch: [{0}][{1}/{2}]  | Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s) | ' \
              'Speed: {speed:.1f} samples/s | Data: {data_time.val:.3f}s ({data_time.avg:.3f}s) | ' \
              'Loss: {loss.val:.5f} ({loss.avg:.5f}) | Accuracy: {acc.val:.3f} ({acc.avg:.3f})'.format(
               epoch, i, len(train_loader), batch_time=batch_time, speed=batch_size/batch_time.val,
               data_time=data_time, loss=losses, acc=acces)
        bar.suffix = msg
        bar.next()

        if i % print_freq == 0:
            train_logger.printWrite(msg, isprint=False)

    bar.finish()
    train_logger.close()
    return losses.avg, acces.avg


# Validater
def validate(cfg, val_loader, model, criterion, device, num_classes=17, print_freq=1, flip=True, shift_hpval=0):
    # Batch data container.
    batch_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # predictions
    val_data_len = val_loader.dataset.__len__()
    all_preds = np.zeros((val_data_len, num_classes, 3), dtype=np.float32)  # e.g. [total_val_data_len, 17, 3]
    all_boxes = np.zeros((val_data_len, 5))  # e.g.[total_val_data_len, 5]
    imgNameList = []
    imgIdList = []

    # Logger
    val_logger = Logger(cfg.LOGGER.VAL_LOGGER, '=====-- Val Logger --=====')

    # visualize tool
    bar = Bar('Validate', max=len(val_loader))

    idx = 0
    end = time.time()
    with torch.no_grad():
        for i, (input_img, target, target_weight, info) in enumerate(val_loader):

            # choose device
            input_img = input_img.to(device)
            target = [tar.to(device, non_blocking=True) for tar in target]
            target_weight = target_weight.to(device, non_blocking=True)

            # prediction
            global_outputs, refine_output = model(input_img)
            # output = outputs[-1].cpu() if isinstance(outputs, list) else outputs.cpu()
            output = refine_output

            # is flip?
            if flip:
                input_img_flip = input_img.flip(3)  # torch.flip(dim)
                outputs_flip = model(input_img_flip)[1]
                output_flip = outputs_flip[-1].cpu() if isinstance(outputs_flip, list) else outputs_flip.cpu()
                # flip back
                flip_pairs = val_loader.dataset.flip_pairs
                output_flip = flip_back(output_flip.numpy(), flip_pairs)
                output_flip = torch.from_numpy(output_flip.copy()).to(device)

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if shift_hpval != 0:
                    output_flip[:, :, :, shift_hpval:] = output_flip.clone()[:, :, :, 0:-shift_hpval]
                output = (output + output_flip) * 0.5

            # Loss
            loss = criterion(output, target[-1], target_weight)
            losses.update(loss.item(), len(val_loader))

            # measure accuracy and record loss
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target[-1].cpu().numpy())
            acces.update(avg_acc, cnt)

            # measure processed time
            batch_time.update(time.time() - end)
            end = time.time()

            # get_final_preds
            c = info['center'].numpy()
            s = info['scale'].numpy()
            preds, maxvals = get_final_preds(output.clone().cpu().numpy(), c, s)

            imgNameList += info['image_name']  # get image name
            imgIdList += info['id'].numpy().tolist()  # get anno's id

            num_images = len(input_img)
            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)  # 200=? pixel_std in data_loader??
            idx += num_images

            msg = 'Test: [{0}/{1}] | Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) | ' \
                  'Loss: {loss.val:.4f} ({loss.avg:.4f}) | Accuracy: {acc.val:.3f} ({acc.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses, acc=acces)
            bar.suffix = msg
            bar.next()

            if i % print_freq == 0:
                val_logger.printWrite(msg, isprint=False)

    bar.finish()
    val_logger.close()
    return losses.avg, acces.avg, {'imgIdList': imgIdList, 'imgNameList': imgNameList,
                                   'all_preds': all_preds, 'all_boxes': all_boxes}







if __name__ == '__main__':
   pass