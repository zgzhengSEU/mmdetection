from mmdet.models.necks.skippafpn import ImprovedPAFPN
import torch


import hiddenlayer as h
from torchviz import make_dot

model = ImprovedPAFPN(
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    use_type='PAFPN_CARAFE_Skip_Parallel_concat',
    # use_type='PAFPN_CARAFE_Skip_Parallel_Old',
    add_extra_convs='on_output',
    concat_kernel_size=1,
    num_outs=5)
x = [torch.randn([1 ,256, 200, 200]), torch.randn([1 ,512, 100, 100]), torch.randn([1 ,1024, 50, 50]), torch.randn([1 ,2048, 25, 25])]
y = model(x)

print(len(y))

vis = make_dot(y, params=dict(list(model.named_parameters())))
vis.format = "png"
vis.view()
# vis_graph = h.build_graph(model, )   # 获取绘制图像的对象
# vis_graph.theme = h.graph.THEMES["blue"].copy()     # 指定主题颜色
# vis_graph.save("./demo1.png")   # 保存图像的路径