# import torch
# from mmcv.ops.carafe import CARAFEPack
# from mmcv.cnn import build_upsample_layer
# # or "from carafe import CARAFEPack"
# upsample_cfg=dict(
#                      type='carafe',
#                      up_kernel=5,
#                      up_group=1,
#                      encoder_kernel=3,
#                      encoder_dilation=1)
# upsample_cfg.update(channels=40, scale_factor=2)
# upsample_module = build_upsample_layer(upsample_cfg)
# # upsample_module.init_weights()

# x = torch.rand(2, 40, 50, 70)
# # model = CARAFEPack(channels=40, scale_factor=2)


# # model = model.cuda()
# # x = x.cuda()
# # upsample_module = upsample_module.cuda()
# out = upsample_module(x)

# print('original shape: ', x.shape)
# print('upscaled shape: ', out.shape)
from mmdet.models.task_modules.prior_generators import AnchorGenerator
self = AnchorGenerator(strides=[4], ratios=[1.], scales=[8.], base_sizes=[9])
all_anchors = self.grid_priors([(1, 1)], device='cpu')
print(all_anchors)