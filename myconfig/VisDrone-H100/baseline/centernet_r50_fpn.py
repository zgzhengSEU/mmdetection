_base_ = [
    '../../../configs/_base_/datasets/visdrone_detection.py',
    '../../../configs/_base_/schedules/schedule_1x.py', '../../../configs/_base_/default_runtime.py'
]

# ======================== wandb & run =========================================================================================

# ===========================================
TAGS = ["centernet"]
GROUP_NAME = "Baseline"
ALGO_NAME = "centernet_r50_fpn"
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

# ==================================================================================================================================================
load_from = ''

model = dict(
    type='CenterNet',
    # use caffe img_norm
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
        style='pytorch',
        init_cfg=dict(type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        # There is a chance to get 40.3 after switching init_cfg,
        # otherwise it is about 39.9~40.1
        init_cfg=dict(type='Caffe2Xavier', layer='Conv2d'),
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='CenterNetUpdateHead',
        num_classes=10,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        hm_min_radius=4,
        hm_min_overlap=0.8,
        more_pos_thresh=0.2,
        more_pos_topk=9,
        soft_weight_on_reg=False,
        loss_cls=dict(
            type='GaussianFocalLoss',
            pos_weight=0.25,
            neg_weight=0.75,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
    ),
    train_cfg=None,
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
        start_factor=0.00025,
        by_epoch=False,
        begin=0,
        end=4000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.01),
    # Experiments show that there is no need to turn on clip_grad.
    paramwise_cfg=dict(norm_decay_mult=0.))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(enable=False, base_batch_size=16)
