_base_ = [
    '../../../configs/_base_/models/visdrone-cascade-rcnn_r50_fpn.py',
    '../../../configs/common/visdrone-lsj-200e_coco-detection.py'
]

# ======================== wandb & run =========================================================================================
TAGS = ["ALL200e","DH", "GA0010", "bifpncarafe","GIOU", "GuidAnchor", "DH"]
GROUP_NAME = "cascade-rcnn"
ALGO_NAME = "cascade-rcnn_r101_bifpncarage_8xb8-amp-lsj-200e_DH_GIOU_GA0010_GuidAnchor"
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

load_from = "https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r101_fpn_mstrain_3x_coco/cascade_mask_rcnn_r101_fpn_mstrain_3x_coco_20210628_165236-51a2d363.pth"
# ==================================================================================================================================

image_size = (1024, 1024)
batch_augments = [dict(type='BatchFixedSizePad', size=image_size)]

# disable allowed_border to avoid potential errors.
# model = dict(
#     data_preprocessor=dict(batch_augments=batch_augments),
#     train_cfg=dict(rpn=dict(allowed_border=-1)))

train_dataloader = dict(batch_size=8, num_workers=8)
# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper = dict(
    type='AmpOptimWrapper',
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(
        type='SGD', lr=0.02 * 4, momentum=0.9, weight_decay=0.00004))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(enable=True, base_batch_size=64)

model = dict(
    data_preprocessor=dict(batch_augments=batch_augments, pad_size_divisor=64),
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
        stage_with_dcn=(False, True, True, True),
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    neck=dict(
        _delete_=True,
        type='BiFPNCarafe',
        num_stages=6,
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0),
    rpn_head=dict(
        _delete_=True,
        type='GARPNHead',
        in_channels=256,
        feat_channels=256,
        approx_anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=8,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        square_anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales=[8],
            strides=[4, 8, 16, 32, 64]),
        anchor_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.07, 0.07, 0.14, 0.14]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.07, 0.07, 0.11, 0.11]),
        loc_filter_thr=0.01,
        loss_loc=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_shape=dict(type='BoundedIoULoss', beta=0.2, loss_weight=1.0),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
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
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True, # GIOULoss
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
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
                    target_stds=[0.0415, 0.0415, 0.0835, 0.0835]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
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
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))]),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            ga_assigner=dict(
                type='ApproxMaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            ga_sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            center_ratio=0.2,
            ignore_ratio=0.5,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_post=1000,
            nms_pre=2000,
            max_per_img=300,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.8,
                    neg_iou_thr=0.8,
                    min_pos_iou=0.8,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_post=1000,
            nms_pre=1000,
            max_per_img=300,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))

