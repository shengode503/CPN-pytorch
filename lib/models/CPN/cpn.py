# Import
import os
import math
import torch
import torch.nn as nn
from collections import OrderedDict


def conv3x3(inplanes, outplanes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(inplanes, outplanes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes,
                 stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class Resnet(nn.Module):
    def __init__(self, block, layers):
        super(Resnet, self).__init__()

        self.inplanes = 64

        # ---------------------------------------------------------------------------

        # ResNet
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x4, x3, x2, x1]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.1),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def init_weights(self, device, pretrained=''):
        print('-------> {0} #'.format("Initial ResNet's Weight with ImageNet Pretrained Model."))
        if os.path.isfile(pretrained):
            print('=> loading pretrained model {}'.format(pretrained))
            checkpoint = torch.load(pretrained, map_location=torch.device(device))
            if isinstance(checkpoint, OrderedDict):
                state_dict = checkpoint
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict_old = checkpoint['state_dict']
                state_dict = OrderedDict()

                # delete 'module.' because it is saved from DataParallel module
                for key in state_dict_old.keys():
                    if key.startswith('module.'):
                        state_dict[key[7:]] = state_dict_old[key]
                    else:
                        state_dict[key] = state_dict_old[key]
            else:
                raise RuntimeError(
                    'No state_dict found in checkpoint file {}'.format(pretrained))
            self.load_state_dict(state_dict, strict=False)
        else:
            print('=> imagenet pretrained model dose not exist')
            print('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')


class GlobalNet(nn.Module):
    def __init__(self, resnet, num_joints, ch_list, output_shape):
        super(GlobalNet, self).__init__()

        self.ch_list = ch_list

        # ResNet
        self.resnet = resnet

        # define layers
        lateral_list, upsample_list, predict_list = [], [], []
        for i in range(len(ch_list)):
            lateral_list.append(self._lateral(ch_list[i]))
            predict_list.append(self._predict(output_shape, num_joints))
            if i != len(ch_list) - 1:
                upsample_list.append(self._upsample())

        self.lateral_list = nn.ModuleList(lateral_list)
        self.upsample_list = nn.ModuleList(upsample_list)
        self.predict_list = nn.ModuleList(predict_list)

        # Initial [lateral, upsample, predict]'s weight.
        print('-------> {0} #'.format("Initial GlobalNet's Weight."))
        print('-------> {0} #'.format('lateral_list'))
        self.lateral_list.apply(self._init_weights)
        print('-------> {0} #'.format('upsample_list'))
        self.upsample_list.apply(self._init_weights)
        print('-------> {0} #'.format('predict_list'))
        self.predict_list.apply(self._init_weights)

    # Lateral
    def _lateral(self, input_size):
        layer = [nn.Conv2d(input_size, 256, kernel_size=1, stride=1, bias=False),
                 nn.BatchNorm2d(256),
                 nn.ReLU(inplace=True)]
        return nn.Sequential(*layer)

    # Up sampling
    def _upsample(self):
        layer = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                 nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
                 nn.BatchNorm2d(256)]
        return nn.Sequential(*layer)

    # Predicting
    def _predict(self, output_shape, num_joints):
        layer = [nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
                 nn.BatchNorm2d(256),
                 nn.ReLU(inplace=True)]
        layer += [nn.Conv2d(256, num_joints, kernel_size=3, stride=1, padding=1, bias=False),
                  nn.Upsample(size=output_shape, mode='bilinear', align_corners=True),
                  nn.BatchNorm2d(num_joints)]
        return nn.Sequential(*layer)

    # initial weight
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            print('=> init {}.weight as normal(0, sqrt(2. / w.h*w.w*w.c))'.format(m))
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            nn.init.normal_(m.weight, std=math.sqrt(2. / n))
            if m.bias is not None:
                print('=> init {}.bias as 0'.format(m))
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            print('=> init {}.weight as 1'.format(m))
            print('=> init {}.bias as 0'.format(m))
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        tiers = len(self.ch_list)

        # ResNet
        x = self.resnet(x)

        global_fms, global_outs, up = [], [], 0
        for i in range(tiers):

            # global_fms (pass to refineNet)
            lateral = self.lateral_list[i](x[i])
            feature = lateral if i == 0 else lateral + up
            global_fms.append(feature)

            # global_outs (for compute loss fn)
            if i != tiers - 1:
                up = self.upsample_list[i](feature)

            feature = self.predict_list[i](feature)
            global_outs.append(feature)

        return global_fms, global_outs


class RefineNet(nn.Module):
    def __init__(self, lateral_channel, out_shape, num_class):
        super(RefineNet, self).__init__()
        # Cascade layers
        self.num_cascade = 4
        self.cascade = nn.ModuleList([self._make_layer(self.num_cascade-1-i, lateral_channel, out_shape
                                                       ) for i in range(self.num_cascade)])

        # Prediction
        self.final_predict = self._predict(lateral_channel*4, num_class)

        # Initial Weight
        print('-------> {0} #'.format("Initial Refine's Weight."))
        self.cascade.apply(self._init_weights)
        self.final_predict.apply(self._init_weights)

    def _make_layer(self, num, input_channel, output_shape):
        planes = 128
        # Change Bottleneck Setting
        Bottleneck.expansion = 2
        downsample = nn.Sequential(
            nn.Conv2d(input_channel, planes * 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes * 2)
        )
        layers = [Bottleneck(input_channel, planes, downsample=downsample) for _ in range(num)]
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        return nn.Sequential(*layers)

    def _predict(self, input_channel, num_class):
        planes = 128
        # Change Bottleneck Setting
        Bottleneck.expansion = 2
        downsample = nn.Sequential(
            nn.Conv2d(input_channel, planes * 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes * 2)
        )
        layers = [Bottleneck(input_channel, planes, downsample=downsample),
                  nn.Conv2d(planes*Bottleneck.expansion, num_class,
                            kernel_size=3, stride=1, padding=1, bias=False),
                  nn.BatchNorm2d(num_class)]
        return nn.Sequential(*layers)

    # initial weight
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            print('=> init {}.weight as normal(0, sqrt(2. / w.h*w.w*w.c))'.format(m))
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            nn.init.normal_(m.weight, std=math.sqrt(2. / n))
            if m.bias is not None:
                print('=> init {}.bias as 0'.format(m))
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            print('=> init {}.weight as 1'.format(m))
            print('=> init {}.bias as 0'.format(m))
            nn.init.constant_(m.weight, 1)


    def forward(self, x):
        refine_fms = [self.cascade[i](x[i]) for i in range(self.num_cascade)]
        out = torch.cat(refine_fms, dim=1)
        out = self.final_predict(out)
        return out


class CPN(nn.Module):
    def __init__(self, output_shape, num_joints, channel_list,  # CPN Setting
                 block_class, layers, model_name,  # ResNet Setting
                 device, is_train=False, is_pretrain=False  # ResNet Setting
                 ):
        super(CPN, self).__init__()

        # Resnet
        resnet = Resnet(block_class, layers)

        # Load PreTrain weight
        if is_train and is_pretrain:
            resnet.init_weights(device, model_name)

        ''' GlobalNet '''
        self.globalnet = GlobalNet(resnet, num_joints, ch_list=channel_list, output_shape=output_shape)

        ''' RefineNet '''
        self.refinenet = RefineNet(channel_list[-1], out_shape=output_shape, num_class=num_joints)

    def forward(self, x):

        # GlobalNet
        global_fms, global_outs = self.globalnet(x)

        # RefineNet
        refine_out = self.refinenet(global_fms)

        return global_outs, refine_out

def get_pose_net(cfg, is_train):

    model_cfg = cfg.MODEL.EXTRA

    resnet_spec = {18: (BasicBlock, [2, 2, 2, 2], 'resnet18-5c106cde.pth'),
                   34: (BasicBlock, [3, 4, 6, 3], 'resnet34-333f7ec4.pth'),
                   50: (Bottleneck, [3, 4, 6, 3], 'resnet50-19c8e357.pth'),
                   101: (Bottleneck, [3, 4, 23, 3], 'resnet101-5d3b4d8f.pth'),
                   152: (Bottleneck, [3, 8, 36, 3], 'resnet152-b121ed2d.pth')}
    block_class, layers, model_name = resnet_spec[model_cfg.RESNET_TYPE]

    model = CPN(output_shape=cfg.MODEL.HEATMAP_SIZE, num_joints=cfg.DATASET.NJOINTS,
                channel_list=model_cfg.CHANNEL_LIST, block_class=block_class, layers=layers,
                model_name=os.path.join(cfg.MODEL.PRETRAIN_RES_ROOT, model_name),
                device=cfg.DEVICE, is_train=is_train, is_pretrain=model_cfg.IS_PRETRAIN)

    return model



if __name__ == '__main__':

    # resnet_spec = {18: (BasicBlock, [2, 2, 2, 2], 'pretrain_resnet/resnet18-5c106cde.pth'),
    #                34: (BasicBlock, [3, 4, 6, 3], 'pretrain_resnet/resnet34-333f7ec4.pth'),
    #                50: (Bottleneck, [3, 4, 6, 3], 'pretrain_resnet/resnet50-19c8e357.pth'),
    #                101: (Bottleneck, [3, 4, 23, 3], 'pretrain_resnet/resnet101-5d3b4d8f.pth'),
    #                152: (Bottleneck, [3, 8, 36, 3], 'pretrain_resnet/resnet152-b121ed2d.pth')}
    #
    # resnet_type = 50
    # block_class, layers, PRETRAIN_MODEL_PATH = resnet_spec[resnet_type]
    #
    # resnet = Resnet(block_class, layers)
    #
    # # Load PreTrain weight
    # is_train = True
    # is_pretrain = True
    # if is_train and is_pretrain:
    #     resnet.init_weights('cpu', PRETRAIN_MODEL_PATH)
    #
    # # Test
    # img = torch.randn((1, 3, 192, 256))
    # out = resnet(img)
    #
    # print(out[0].shape)
    # print(out[1].shape)
    # print(out[2].shape)
    # print(out[3].shape)

    # --------------------------------------------------

    # # ResNet
    # resnet_spec = {18: (BasicBlock, [2, 2, 2, 2], 'pretrain_resnet/resnet18-5c106cde.pth'),
    #                34: (BasicBlock, [3, 4, 6, 3], 'pretrain_resnet/resnet34-333f7ec4.pth'),
    #                50: (Bottleneck, [3, 4, 6, 3], 'pretrain_resnet/resnet50-19c8e357.pth'),
    #                101: (Bottleneck, [3, 4, 23, 3], 'pretrain_resnet/resnet101-5d3b4d8f.pth'),
    #                152: (Bottleneck, [3, 8, 36, 3], 'pretrain_resnet/resnet152-b121ed2d.pth')}
    #
    # resnet_type = 50
    # block_class, layers, PRETRAIN_MODEL_PATH = resnet_spec[resnet_type]
    #
    # resnet = Resnet(block_class, layers)
    #
    # # Load PreTrain weight
    # is_train = True
    # is_pretrain = True
    # if is_train and is_pretrain:
    #     resnet.init_weights('cpu', PRETRAIN_MODEL_PATH)
    #
    # # GlobalNet
    # globalnet = GlobalNet(resnet, num_joints=17,
    #                       ch_list=[2048, 1024, 512, 256],
    #                       output_shape=(96, 72))
    #
    # img = torch.randn((1, 3, 192, 256))
    # global_fms, global_outs = globalnet(img)
    #
    #
    # # -------------------------------------------
    # data_shape = (192, 256)  # height, width
    # output_shape = (72, 96)  # height, width
    #
    # ch_list = [2048, 1024, 512, 256]
    # refinenet = RefineNet(ch_list[-1], output_shape, num_class=17)
    #
    # kevingg = refinenet(global_fms)
    #


    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    from lib.config import cfg
    cfg.freeze()

    model = get_pose_net(cfg, is_train=False)

    img = torch.randn((1, 3, 256, 256))
    target = torch.randn((1, 17, 64, 64))
    loss_w = torch.FloatTensor([1] * 17)[None]

    global_outputs, refine_output = model(img)

    # Loss
    from lib.models.Loss.mseLoss import JointsOHKMMSELoss, JointsMSELoss

    criterion = JointsMSELoss(use_target_weight=True).to(cfg.DEVICE)
    criterion_ohkm = JointsOHKMMSELoss(use_target_weight=True).to(cfg.DEVICE)

    # GlobalNet Loss
    loss = 0
    for global_output in global_outputs:
        loss += criterion(global_output, target, loss_w)

    # RefineNet Loss
    loss += criterion_ohkm(refine_output, target, loss_w)