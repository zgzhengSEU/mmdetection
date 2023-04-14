_base_ = [
    '../../../configs/_base_/models/visdrone-cascade-rcnn_r50_fpn.py',
    '../../../configs/_base_/datasets/visdrone_detection.py',
    '../../../configs/_base_/schedules/schedule_1x.py', '../../../configs/_base_/default_runtime.py'
]

# ======================== wandb & run =========================================================================================

# ===========================================
TAGS = ["casc_x50-32x4d_fpn_1x", 'SCPAFPN', 'tinyanchor']
GROUP_NAME = "cascade-rcnn"
ALGO_NAME = "cascade-rcnn_x50-32x4d_fpn_1x_tinyanchor_SCPAFPN"
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


checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnext/resnext50_32x4d_b32x8_imagenet_20210429-56066e27.pth'  # noqa
model = dict(
    backbone=dict(
        type='ResNeXt',
        groups=32,
        base_width=4,
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    neck=dict(
        type='ImprovedPAFPN',
        use_type='PAFPN_CARAFE_Skip_Parallel_Old',
        add_extra_convs='on_output',
        reduce_kernel_size=3,
        concat_kernel_size=1,
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
            ratios=[0.333, 0.5, 1.0, 2.0, 3.0])))

optim_wrapper = dict(
    optimizer=dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0., bypass_duplicate=True))


