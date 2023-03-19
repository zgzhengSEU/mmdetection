_base_ = [
    '../../../configs/_base_/models/visdrone-cascade-rcnn_r50_fpn.py',
    '../../../configs/_base_/datasets/visdrone_detection.py',
    '../../../configs/_base_/schedules/schedule_1x.py', '../../../configs/_base_/default_runtime.py'
]
# ======================== wandb & run =========================================================================================
TAGS = ["cascade-rcnn_convnextt_bifpn_giou_DH"]
GROUP_NAME = "cascade-rcnn"
ALGO_NAME = "cascade-rcnn_convnextt-p4-w7_bifpn_giou_amp-ms-crop-3x_DH"
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
# ======================== wandb & run =========================================================================================

# =============== datasets ======================================================================================================
# Batch size of a single GPU during training
train_batch_size_per_gpu = 2
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 2
# Batch size of a single GPU during valing
val_batch_size_per_gpu = 1
# Worker to pre-fetch data for each single GPU during valing
val_num_workers = 2
# Batch size of a single GPU during valing
test_batch_size_per_gpu = 1
# Worker to pre-fetch data for each single GPU during valing
test_num_workers = 2


val_dataloader = dict(batch_size=val_batch_size_per_gpu, num_workers=val_num_workers)
test_dataloader = dict(batch_size=test_batch_size_per_gpu, num_workers=test_num_workers)


# TODO: delete custom_imports after mmcls supports auto import
# please install mmcls>=1.0
# import mmcls.models to trigger register_module in mmcls
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'  # noqa

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmcls.ConvNeXt',
        arch='tiny',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    neck=dict(
        _delete_=True,
        type='BiFPN',
        num_stages=3,
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        start_level=0),
    roi_head=dict(
        type='CascadeDoubleHeadRoIHead',
        reg_roi_scale_factor=1.3,    
        bbox_head=[
            dict(
                type='DoubleConvFCBBoxHead',
                num_convs=5,
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
                reg_class_agnostic=False, #
                reg_decoded_bbox=True, # GIOULoss
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='DoubleConvFCBBoxHead',
                num_convs=5,
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
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='DoubleConvFCBBoxHead',
                num_convs=5,
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
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
    ]))

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[[
            dict(
                type='RandomChoiceResize',
                scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                        (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                        (736, 1333), (768, 1333), (800, 1333)],
                keep_ratio=True)
        ],
                    [
                        dict(
                            type='RandomChoiceResize',
                            scales=[(400, 1333), (500, 1333), (600, 1333)],
                            keep_ratio=True),
                        dict(
                            type='RandomCrop',
                            crop_type='absolute_range',
                            crop_size=(384, 600),
                            allow_negative_crop=True),
                        dict(
                            type='RandomChoiceResize',
                            scales=[(480, 1333), (512, 1333), (544, 1333),
                                    (576, 1333), (608, 1333), (640, 1333),
                                    (672, 1333), (704, 1333), (736, 1333),
                                    (768, 1333), (800, 1333)],
                            keep_ratio=True)
                    ]]),
    dict(type='PackDetInputs')
]

train_dataloader = dict(batch_size=train_batch_size_per_gpu, num_workers=train_num_workers, dataset=dict(pipeline=train_pipeline))
max_epochs = 36
train_cfg = dict(max_epochs=max_epochs)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1)
]

# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper = dict(
    type='AmpOptimWrapper',
    constructor='LearningRateDecayOptimizerConstructor',
    paramwise_cfg={
        'decay_rate': 0.7,
        'decay_type': 'layer_wise',
        'num_layers': 6
    },
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0002,
        betas=(0.9, 0.999),
        weight_decay=0.05))
