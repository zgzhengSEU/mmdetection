_base_ = [
    '../../../configs/_base_/models/visdrone-cascade-rcnn_r50_fpn.py',
    '../../../configs/_base_/datasets/visdrone_detection.py',
    '../../../configs/_base_/schedules/schedule_20e.py', '../../../configs/_base_/default_runtime.py'
]


# ======================== wandb & run =========================================================================================
TAGS = ["casc_r50_fpn_20e","DH", "GA0010", "bifpn"]
GROUP_NAME = "cascade-rcnn"
ALGO_NAME = "cascade-rcnn_r50_bifpn_20e_DH_GA0010"
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

load_from = "https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_20e_coco/cascade_rcnn_r50_fpn_20e_coco_bbox_mAP-0.41_20200504_175131-e9872a90.pth"

# =============== datasets ======================================================================================================
# Batch size of a single GPU during training
train_batch_size_per_gpu = 1
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 1
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

# =============== model ========================================================================================================
model = dict(
    backbone=dict(
        plugins=[
            dict(
                cfg=dict(
                    type='GeneralizedAttention',
                    spatial_range=-1,
                    num_heads=8,
                    attention_type='0010',
                    kv_stride=2),
                stages=(False, False, True, True),
                position='after_conv2')
        ],
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        _delete_=True,
        type='BiFPN',
        num_stages=6,
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0),
    roi_head=dict(
        type='CascadeDoubleHeadRoIHead',
        reg_roi_scale_factor=1.3,
        bbox_head=[
            dict(
                type='DoubleConvFCBBoxHead',
                num_convs=4,
                num_fcs=2,
                in_channels=256,
                conv_out_channels=1024,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=2.0)),
            dict(
                type='DoubleConvFCBBoxHead',
                num_convs=4,
                num_fcs=2,
                in_channels=256,
                conv_out_channels=1024,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=2.0)),
            dict(
                type='DoubleConvFCBBoxHead',
                num_convs=4,
                num_fcs=2,
                in_channels=256,
                conv_out_channels=1024,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=2.0))]))

# ===================== optimizer ======================================================================================================
# optim_wrapper = dict(
#     optimizer=dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.05),
#     paramwise_cfg=dict(norm_decay_mult=0., bypass_duplicate=True))