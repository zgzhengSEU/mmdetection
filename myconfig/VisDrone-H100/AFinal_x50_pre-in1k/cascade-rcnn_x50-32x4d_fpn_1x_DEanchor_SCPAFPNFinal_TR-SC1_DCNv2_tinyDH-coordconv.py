_base_ = [
    '../../../configs/_base_/models/visdrone-cascade-rcnn_r50_fpn.py',
    '../../../configs/_base_/datasets/visdrone_detection.py',
    '../../../configs/_base_/schedules/schedule_1x.py', '../../../configs/_base_/default_runtime.py'
]

# ======================== wandb & run =========================================================================================

# ===========================================
TAGS = ["cascade-rcnn_x50-32x4d_fpn_1x", 'tinyDHCC', 'DEanchor', 'SCPAFPNFinal', 'TR-SC1', 'DCNv2']
GROUP_NAME = "cascade-rcnn-final"
ALGO_NAME = "cascade-rcnn_x50-32x4d_fpn_1x_DEanchor_SCPAFPNFinal_TR-SC1_DCNv2_tinyDH-coordconv"
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
train_batch_size_per_gpu = 8
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
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth'
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnext/resnext50_32x4d_b32x8_imagenet_20210429-56066e27.pth'  # noqa
model = dict(
    backbone=dict(
        type='ResNeXt',
        groups=32,
        base_width=4,
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint),
        plugins=[
            dict(
                cfg=dict(
                    type='SpatialTR',
                    num_heads=8,
                    q_stride=2,
                    kv_stride=2),
                stages=(False, False, True, True),
                position='after_conv2'),
            dict(
                cfg=dict(
                    type='ChannelTR',
                    num_heads=1,
                    kerner_size=1,
                    reduce=4,
                    use_in_conv=True,
                    use_out_conv=True,
                    use_downsample=False # set pad_size_divisor=128 when True
                ),
                stages=(False, False, True, True),
                position='after_conv2')
        ],
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='ImprovedPAFPN',
        use_type='SCPAFPNFinal',
        add_extra_convs='on_output',
        reduce_kernel_size=1,
        concat_kernel_size=1,
        use_concat_conv=True,
        fpn_kernel_size=3,
        pafpn_kernel_size=3,
        norm_cfg=None,
        upsample_cfg=dict(
            type='carafe',
            up_kernel=5,
            up_group=1,
            encoder_kernel=3,
            encoder_dilation=1,
            compressed_channels=64)),
    rpn_head=dict(
        anchor_generator=dict(
            scales=[4],
            # base_sizes = [6, 12, 24, 48, 96],
            ratios=[0.371, 0.605, 1.0, 1.653, 2.696])),
    roi_head=dict(
        type='CascadeDoubleHeadRoIHead', # new
        reg_roi_scale_factor=1.3, # new
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='DoubleConvFCBBoxHead',
                use_coordconv= True,
                num_convs=4, # new
                num_fcs=2, # new
                conv_out_channels=1024, # new
                fc_out_channels=1024,
                in_channels=256,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False, # change
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='DoubleConvFCBBoxHead',
                use_coordconv= True,
                num_convs=3, # new
                num_fcs=2, # new
                conv_out_channels=1024, # new
                fc_out_channels=1024,
                in_channels=256,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False, # change
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='DoubleConvFCBBoxHead',
                use_coordconv= True,
                num_convs=2, # new
                num_fcs=2, # new
                conv_out_channels=1024, # new
                fc_out_channels=1024,
                in_channels=256,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False, # change
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0))
        ]) 
    )

optim_wrapper = dict(
    optimizer=dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0., bypass_duplicate=True))


