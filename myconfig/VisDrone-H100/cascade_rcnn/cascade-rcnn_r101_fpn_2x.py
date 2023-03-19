_base_ = [
    '../../../configs/_base_/models/visdrone-cascade-rcnn_r50_fpn.py',
    '../../../configs/_base_/datasets/visdrone_detection.py',
    '../../../configs/_base_/schedules/schedule_2x.py', '../../../configs/_base_/default_runtime.py'
]


# ======================== wandb & run =========================================================================================
# bsub -J cascade-rcnn_r101_fpn_2x -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3;module load cuda-11.6;module load gcc-9.3.0;source activate mmdet3;cd mmdet3;python3 tools/train.py myconfig/VisDrone-seu/cascade_rcnn/cascade-rcnn_r101_fpn_2x.py"
# ===========================================
TAGS = ["r101", "2x"]
GROUP_NAME = "cascade-rcnn"
ALGO_NAME = "cascade-rcnn_r101_fpn_2x"
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

#load_from = "https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r101_fpn_20e_coco/cascade_rcnn_r101_fpn_20e_coco_bbox_mAP-0.425_20200504_231812-5057dcc5.pth"

# =============== datasets ======================================================================================================
# Batch size of a single GPU during training
train_batch_size_per_gpu = 16 # 4->18G  8->24G 16->26G
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

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))

"""
==============================
Use size divisor set input shape from (1080, 1920) to (768, 1344)
==============================
Compute type: dataloader: load a picture from the dataset
Input shape: (768, 1344)
Flops: 0.312T
Params: 88.172M
==============================
"""
