_base_ = [
    '../../../configs/_base_/models/BJH-cascade-rcnn_r50_fpn.py',
    '../../../configs/_base_/datasets/BJH_detection.py',
    '../../../configs/_base_/schedules/schedule_2x.py', '../../../configs/_base_/default_runtime.py'
]

# ======================== wandb & run =========================================================================================

# ===========================================
TAGS = ["casc_r50_fpn_2x", 'mmpretrain']
GROUP_NAME = "cascade-rcnn"
ALGO_NAME = "cascade-rcnn_r50_fpn_2x"
DATASET_NAME = "BJHDrone"

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

checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth'  # mmpretrain-resnet50
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)))