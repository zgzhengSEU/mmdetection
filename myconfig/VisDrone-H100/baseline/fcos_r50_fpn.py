_base_ = [
    '../../../configs/_base_/datasets/visdrone_detection.py',
    '../../../configs/_base_/schedules/schedule_1x.py', '../../../configs/_base_/default_runtime.py' 
]


# ======================== wandb & run =========================================================================================

# ===========================================
TAGS = ["fcos"]
GROUP_NAME = "Baseline"
ALGO_NAME = "fcos_r50_fpn"
DATASET_NAME = "VisDrone"

Wandb_init_kwargs = dict(
    project=DATASET_NAME,
    group=GROUP_NAME,
    name=ALGO_NAME,
    tags=TAGS,
    resume="allow",
    # id="",
    allow_val_change=True
)
visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend', init_kwargs=Wandb_init_kwargs)])

# ==========================================
import datetime as dt
NOW_TIME = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
work_dir = f"work_dirs/{DATASET_NAME}/{ALGO_NAME}/{NOW_TIME}"

# =============== datasets ======================================================================================================
# Batch size of a single GPU during training
train_batch_size_per_gpu = 16
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 8
# Batch size of a single GPU during valing
val_batch_size_per_gpu = 1
# Worker to pre-fetch data for each single GPU during valing
val_num_workers = 2
# Batch size of a single GPU during valing
test_batch_size_per_gpu = 1
# Worker to pre-fetch data for each single GPU during valing
test_num_workers = 2

train_dataloader = dict(batch_size=train_batch_size_per_gpu, num_workers=train_num_workers)
val_dataloader = dict(batch_size=val_batch_size_per_gpu, num_workers=val_num_workers)
test_dataloader = dict(batch_size=test_batch_size_per_gpu, num_workers=test_num_workers)

# ==================================================================================================================================================
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco-0a0d75a8.pth'
# model settings
model = dict(
    type='FCOS',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        center_sampling=True,
        conv_bias=True,
        num_classes=10,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # testing settings
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3.0,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]
# optimizer
optim_wrapper = dict(
    optimizer=dict(lr=0.01),
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
    clip_grad=None)