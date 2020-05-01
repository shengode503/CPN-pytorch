A Pytorch implementation of "Cascaded Pyramid Network for Multi-Person Pose Estimation"
(https://arxiv.org/abs/1711.07319).

Part of code referenced from :
'https://github.com/GengDavid/pytorch-cpn' &
'https://github.com/HRNet'

# Training:

     1.  Download the ImageNet pre-trained Resnet weight from https://pytorch.org/docs/stable/torchvision/models.html.
         and place into 'pretrain_model/pretrain_resnet/'.
         
     2.  Make sure the training & Model setting in 'experiment/experiment_0.yaml'.
     
     3.  Download the coco dataset from 'http://cocodataset.org/#home'. and place into 'dataset/'.
         Install the cocoapi (https://github.com/cocodataset/cocoapi).
     
     4.  mkdir 'result/check_result',
               'result/checkpoints',
               'result/logs'
     
     5.  Run " training.py "
        
# Evaluation:
     0.  Make sure the 'model_best.pth' is in 'result/checkpoints/'.
     
     1.  Run " evaluation.py ". 

# Inference:
     0.  Make sure the 'model_best.pth' is in 'result/checkpoints/'.
     
     1.  First we need a person detector. (maybe you can use mmdetection or detectron.)
         # mmdetection: 'https://github.com/open-mmlab/mmdetection'
         # detectron: 'https://github.com/facebookresearch/detectron2'
         
     2.  Put input image into 'dataset/test_data/'.
     
     3.  Run " inference.py ". 






