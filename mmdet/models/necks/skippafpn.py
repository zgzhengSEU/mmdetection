
import torch.nn as nn
from mmcv.cnn import ConvModule, build_upsample_layer
from mmcv.ops.carafe import CARAFEPack
from mmengine.model import BaseModule, ModuleList, xavier_init
from mmdet.registry import MODELS
from typing import List, Tuple, Union
from torch import Tensor
import torch
from mmdet.utils import ConfigType, MultiConfig, OptConfigType


@MODELS.register_module()
class PAFPN_CARAFE_Skip_Parallel(BaseModule):
    """FPN_CARAFE is a more flexible implementation of FPN. It allows more
    choice for upsample methods during the top-down pathway.

    It can reproduce the performance of ICCV 2019 paper
    CARAFE: Content-Aware ReAssembly of FEatures
    Please refer to https://arxiv.org/abs/1905.02188 for more details.

    Args:
        in_channels (list[int]): Number of channels for each input feature map.
        out_channels (int): Output channels of feature pyramids.
        num_outs (int): Number of output stages.
        start_level (int): Start level of feature pyramids.
            (Default: 0)
        end_level (int): End level of feature pyramids.
            (Default: -1 indicates the last level).
        norm_cfg (dict): Dictionary to construct and config norm layer.
        activate (str): Type of activation function in ConvModule
            (Default: None indicates w/o activation).
        order (dict): Order of components in ConvModule.
        upsample (str): Type of upsample layer.
        upsample_cfg (dict): Dictionary to construct and config upsample layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 norm_cfg=None,
                 act_cfg=None,
                 add_extra_convs: Union[bool, str] = 'on_output',
                 order=('conv', 'norm', 'act'),
                 upsample_cfg=dict(type='nearest'),
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(PAFPN_CARAFE_Skip_Parallel, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_bias = norm_cfg is None
        self.upsample_cfg = upsample_cfg.copy()
        self.upsample = self.upsample_cfg.get('type')
        self.relu = nn.ReLU(inplace=False)
        self.add_extra_convs = add_extra_convs
        self.order = order
        assert order in [('conv', 'norm', 'act'), ('act', 'conv', 'norm')]

        assert self.upsample in [
            'nearest', 'bilinear', 'deconv', 'pixel_shuffle', 'carafe', None
        ]
        if self.upsample in ['deconv', 'pixel_shuffle']:
            assert hasattr(
                self.upsample_cfg,
                'upsample_kernel') and self.upsample_cfg.upsample_kernel > 0
            self.upsample_kernel = self.upsample_cfg.pop('upsample_kernel')

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = ModuleList()
        self.fpn_convs = ModuleList()
        self.upsample_modules = ModuleList()
        self.reduce_convs = nn.ModuleList()
        self.parallel_reduce_convs = nn.ModuleList()
        # (0, 4) 一共4个reduce_conv, 3个 fpn_conv、upsample
        for i in range(self.start_level, self.backbone_end_level):
            reduce_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                norm_cfg=norm_cfg,
                bias=self.with_bias,
                act_cfg=act_cfg,
                inplace=False)
            if i != self.backbone_end_level - 1:
                if i != 0:
                    parallel_reduce_conv = ConvModule(
                        in_channels[i],
                        out_channels,
                        1,
                        norm_cfg=norm_cfg,
                        bias=self.with_bias,
                        act_cfg=act_cfg,
                        inplace=False)
                    self.parallel_reduce_convs.append(parallel_reduce_conv)
                fpn_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=self.with_bias,
                    act_cfg=act_cfg,
                    inplace=False,
                    order=self.order)
                upsample_cfg_ = self.upsample_cfg.copy()
                if self.upsample == 'deconv':
                    upsample_cfg_.update(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=self.upsample_kernel,
                        stride=2,
                        padding=(self.upsample_kernel - 1) // 2,
                        output_padding=(self.upsample_kernel - 1) // 2)
                elif self.upsample == 'pixel_shuffle':
                    upsample_cfg_.update(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        scale_factor=2,
                        upsample_kernel=self.upsample_kernel)
                elif self.upsample == 'carafe':
                    upsample_cfg_.update(channels=out_channels, scale_factor=2)
                    print(f'[USE CARAFE upsample op]')
                else:
                    # suppress warnings
                    align_corners = (None
                                     if self.upsample == 'nearest' else False)
                    upsample_cfg_.update(
                        scale_factor=2,
                        mode=self.upsample,
                        align_corners=align_corners)
                upsample_module = build_upsample_layer(upsample_cfg_)
                self.upsample_modules.append(upsample_module)
                self.fpn_convs.append(fpn_conv)
            self.reduce_convs.append(reduce_conv)

        # add extra conv layers (e.g., RetinaNet)
        # 5 - 4 + 0 = 1
        extra_out_levels = (
            num_outs - self.backbone_end_level + self.start_level)
        if extra_out_levels >= 1:
            for i in range(extra_out_levels):
                in_channels = (
                    # self.in_channels[3]
                    self.in_channels[self.backbone_end_level -
                                     1] if i == 0 else out_channels)
                extra_reduce_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    norm_cfg=norm_cfg,
                    bias=self.with_bias,
                    act_cfg=act_cfg,
                    inplace=False,
                    order=self.order)
                self.reduce_convs.append(extra_reduce_conv)

        # add extra bottom up pathway
        self.downsample_convs = nn.ModuleList()
        self.skip_downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            d_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            pafpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            if i != self.backbone_end_level - 1:
                skip_d_conv = ConvModule(
                    out_channels,
                    out_channels,
                    5,
                    stride=4,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.skip_downsample_convs.append(skip_d_conv)
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)
    # default init_weights for conv(msra) and norm in ConvModule

    def init_weights(self):
        """Initialize the weights of module."""
        super(PAFPN_CARAFE_Skip_Parallel, self).init_weights()
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                xavier_init(m, distribution='uniform')
        for m in self.modules():
            if isinstance(m, CARAFEPack):
                m.init_weights()

    def slice_as(self, src, dst):
        """Slice ``src`` as ``dst``

        Note:
            ``src`` should have the same or larger size than ``dst``.

        Args:
            src (torch.Tensor): Tensors to be sliced.
            dst (torch.Tensor): ``src`` will be sliced to have the same
                size as ``dst``.

        Returns:
            torch.Tensor: Sliced tensor.
        """
        assert (src.size(2) >= dst.size(2)) and (src.size(3) >= dst.size(3))
        if src.size(2) == dst.size(2) and src.size(3) == dst.size(3):
            return src
        else:
            return src[:, :, :dst.size(2), :dst.size(3)]

    def tensor_add(self, a, b):
        """Add tensors ``a`` and ``b`` that might have different sizes."""
        if a.size() == b.size():
            c = a + b
        else:
            c = a + self.slice_as(b, a)
        return c

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build reduce out
        reduce_outs = []
        parallel_reduce_outs = []
        for i, reduce_conv in enumerate(self.reduce_convs):
            if i <= self.backbone_end_level - self.start_level:
                input = inputs[min(i + self.start_level, len(inputs) - 1)]
            else:
                input = reduce_outs[-1]
            reduce_out = reduce_conv(input)
            reduce_outs.append(reduce_out)
            if i > 0 and i < self.backbone_end_level - 1:
                parallel_reduce_out = self.parallel_reduce_convs[i - 1](input)
                parallel_reduce_outs.append(parallel_reduce_out)

        # build top-down path
        # 3, 2, 1 更新 reduce_outs[2, 1, 0]
        inner_outs = [reduce_outs[-2], reduce_outs[-1]]
        for i in range(self.backbone_end_level - 1, 0, -1):
            feat_high_small = inner_outs[0]
            feat_low_large = reduce_outs[i - 1]
            upsample_feat = self.upsample_modules[i - 1](feat_high_small)
            inner_out = self.fpn_convs[i -
                                       1](self.tensor_add(feat_low_large, upsample_feat))
            inner_outs.insert(0, inner_out)

        # build bottom-up path\
        outs = [inner_outs[0]]
        for i in range(1, self.num_outs):
            feat_low_large = outs[-1]
            feat_high_small = inner_outs[i]
            downsample_feat = self.downsample_convs[i - 1](feat_low_large)
            if i == 1:
                feat_parallel = parallel_reduce_outs[i- 1]
                out = self.pafpn_convs[i - 1](self.tensor_add(self.tensor_add(
                    downsample_feat, feat_high_small), feat_parallel))
                outs.append(out)
            elif i == 2:
                feat_parallel = parallel_reduce_outs[i - 1]
                skip_feat_low_large = outs[-2]
                skip_downsample_feat = self.skip_downsample_convs[i - 2](
                    skip_feat_low_large)
                out = self.pafpn_convs[i - 1](self.tensor_add(self.tensor_add(self.tensor_add(
                    skip_downsample_feat, downsample_feat), feat_high_small), feat_parallel))
                outs.append(out)
            elif i == 3:
                skip_feat_low_large = outs[-2]
                skip_downsample_feat = self.skip_downsample_convs[i - 2](
                    skip_feat_low_large)
                out = self.pafpn_convs[i - 1](self.tensor_add(self.tensor_add(
                    skip_downsample_feat, feat_high_small), downsample_feat))
                outs.append(out)
            elif i == 4:
                if self.add_extra_convs == 'on_reduce':
                    skip_feat_low_large = outs[-2]
                    skip_downsample_feat = self.skip_downsample_convs[i - 2](
                        skip_feat_low_large)
                    out = self.pafpn_convs[i - 1](self.tensor_add(self.tensor_add(
                        skip_downsample_feat, feat_high_small), downsample_feat))
                    outs.append(out)
                elif self.add_extra_convs == 'on_output':
                    skip_feat_low_large = outs[-2]
                    skip_downsample_feat = self.skip_downsample_convs[i - 2](
                        skip_feat_low_large)
                    out = self.pafpn_convs[i - 1](self.tensor_add(
                        skip_downsample_feat, downsample_feat))
                    outs.append(out)
                elif self.add_extra_convs == 'on_reduce_wo_output':
                    skip_feat_low_large = outs[-2]
                    skip_downsample_feat = self.skip_downsample_convs[i - 2](
                        skip_feat_low_large)
                    out = self.pafpn_convs[i - 1](self.tensor_add(
                        skip_downsample_feat, feat_high_small))
                    outs.append(out)
                else:
                    raise NotImplementedError(
                        f'SKIPPAFPN {i + 1} layer are not implemented !')

        return tuple(outs)

@MODELS.register_module()
class PAFPN_CARAFE_Skip_Parallel_concat(BaseModule):
    """FPN_CARAFE is a more flexible implementation of FPN. It allows more
    choice for upsample methods during the top-down pathway.

    It can reproduce the performance of ICCV 2019 paper
    CARAFE: Content-Aware ReAssembly of FEatures
    Please refer to https://arxiv.org/abs/1905.02188 for more details.

    Args:
        in_channels (list[int]): Number of channels for each input feature map.
        out_channels (int): Output channels of feature pyramids.
        num_outs (int): Number of output stages.
        start_level (int): Start level of feature pyramids.
            (Default: 0)
        end_level (int): End level of feature pyramids.
            (Default: -1 indicates the last level).
        norm_cfg (dict): Dictionary to construct and config norm layer.
        activate (str): Type of activation function in ConvModule
            (Default: None indicates w/o activation).
        order (dict): Order of components in ConvModule.
        upsample (str): Type of upsample layer.
        upsample_cfg (dict): Dictionary to construct and config upsample layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 concat_kernel_size=3,
                 start_level=0,
                 end_level=-1,
                 norm_cfg=None,
                 act_cfg=None,
                 add_extra_convs: Union[bool, str] = 'on_reduce',
                 order=('conv', 'norm', 'act'),
                 upsample_cfg=dict(type='nearest'),
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(PAFPN_CARAFE_Skip_Parallel_concat, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_bias = norm_cfg is None
        self.upsample_cfg = upsample_cfg.copy()
        self.upsample = self.upsample_cfg.get('type')
        self.relu = nn.ReLU(inplace=False)
        self.concat_kernel_size = concat_kernel_size
        self.order = order
        self.add_extra_convs = add_extra_convs
        assert order in [('conv', 'norm', 'act'), ('act', 'conv', 'norm')]

        assert self.upsample in [
            'nearest', 'bilinear', 'deconv', 'pixel_shuffle', 'carafe', None
        ]
        if self.upsample in ['deconv', 'pixel_shuffle']:
            assert hasattr(
                self.upsample_cfg,
                'upsample_kernel') and self.upsample_cfg.upsample_kernel > 0
            self.upsample_kernel = self.upsample_cfg.pop('upsample_kernel')

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = ModuleList()
        self.fpn_convs = ModuleList()
        self.upsample_modules = ModuleList()
        self.reduce_convs = nn.ModuleList()
        self.parallel_reduce_convs = nn.ModuleList()
        # (0, 4) 一共4个reduce_conv, 3个fpn_conv、upsample, 2个parallel_reduce_conv
        for i in range(self.start_level, self.backbone_end_level):
            reduce_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                norm_cfg=norm_cfg,
                bias=self.with_bias,
                act_cfg=act_cfg,
                inplace=False)
            if i != self.backbone_end_level - 1:
                if i != 0:
                    parallel_reduce_conv = ConvModule(
                        in_channels[i],
                        out_channels,
                        1,
                        norm_cfg=norm_cfg,
                        bias=self.with_bias,
                        act_cfg=act_cfg,
                        inplace=False)
                    self.parallel_reduce_convs.append(parallel_reduce_conv)
                fpn_conv = ConvModule(
                    out_channels * 2,
                    out_channels,
                    self.concat_kernel_size,
                    padding=1 if self.concat_kernel_size == 3 else 0,
                    norm_cfg=self.norm_cfg,
                    bias=self.with_bias,
                    act_cfg=act_cfg,
                    inplace=False)
                upsample_cfg_ = self.upsample_cfg.copy()
                if self.upsample == 'deconv':
                    upsample_cfg_.update(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=self.upsample_kernel,
                        stride=2,
                        padding=(self.upsample_kernel - 1) // 2,
                        output_padding=(self.upsample_kernel - 1) // 2)
                elif self.upsample == 'pixel_shuffle':
                    upsample_cfg_.update(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        scale_factor=2,
                        upsample_kernel=self.upsample_kernel)
                elif self.upsample == 'carafe':
                    upsample_cfg_.update(channels=out_channels, scale_factor=2)
                    print(f'[USE CARAFE upsample op]')
                else:
                    # suppress warnings
                    align_corners = (None
                                     if self.upsample == 'nearest' else False)
                    upsample_cfg_.update(
                        scale_factor=2,
                        mode=self.upsample,
                        align_corners=align_corners)
                upsample_module = build_upsample_layer(upsample_cfg_)
                self.upsample_modules.append(upsample_module)
                self.fpn_convs.append(fpn_conv)
            self.reduce_convs.append(reduce_conv)

        # add extra conv layers (e.g., RetinaNet)
        # 5 - 4 + 0 = 1
        extra_out_levels = (
            num_outs - self.backbone_end_level + self.start_level)
        if extra_out_levels >= 1:
            for i in range(extra_out_levels):
                in_channels = (
                    # self.in_channels[3]
                    self.in_channels[self.backbone_end_level -
                                     1] if i == 0 else out_channels)
                extra_reduce_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    norm_cfg=norm_cfg,
                    bias=self.with_bias,
                    act_cfg=act_cfg,
                    inplace=False,
                    order=self.order)
                self.reduce_convs.append(extra_reduce_conv)

        # add extra bottom up pathway
        self.downsample_convs = nn.ModuleList()
        self.skip_downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        pafpn_inchannels = [3, 4, 3, 3 if self.add_extra_convs == 'on_reduce' else 2]
        for i in range(self.start_level, self.backbone_end_level):
            d_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            pafpn_conv = ConvModule(
                out_channels * pafpn_inchannels[i],
                out_channels,
                self.concat_kernel_size,
                padding=1 if self.concat_kernel_size == 3 else 0,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            if i != self.backbone_end_level - 1:
                skip_d_conv = ConvModule(
                    out_channels,
                    out_channels,
                    5,
                    stride=4,
                    padding=2,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.skip_downsample_convs.append(skip_d_conv)
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)
    # default init_weights for conv(msra) and norm in ConvModule

    def init_weights(self):
        """Initialize the weights of module."""
        super(PAFPN_CARAFE_Skip_Parallel_concat, self).init_weights()
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                xavier_init(m, distribution='uniform')
        for m in self.modules():
            if isinstance(m, CARAFEPack):
                m.init_weights()

    def slice_as(self, src, dst):
        """Slice ``src`` as ``dst``

        Note:
            ``src`` should have the same or larger size than ``dst``.

        Args:
            src (torch.Tensor): Tensors to be sliced.
            dst (torch.Tensor): ``src`` will be sliced to have the same
                size as ``dst``.

        Returns:
            torch.Tensor: Sliced tensor.
        """
        assert (src.size(2) >= dst.size(2)) and (src.size(3) >= dst.size(3))
        if src.size(2) == dst.size(2) and src.size(3) == dst.size(3):
            return src
        else:
            return src[:, :, :dst.size(2), :dst.size(3)]

    def tensor_add(self, a, b):
        """Add tensors ``a`` and ``b`` that might have different sizes."""
        if a.size() == b.size():
            c = a + b
        else:
            c = a + self.slice_as(b, a)
        return c

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build reduce out
        reduce_outs = []
        parallel_reduce_outs = []
        for i, reduce_conv in enumerate(self.reduce_convs):
            if i <= self.backbone_end_level - self.start_level:
                input = inputs[min(i + self.start_level, len(inputs) - 1)]
            else:
                input = reduce_outs[-1]
            reduce_out = reduce_conv(input)
            reduce_outs.append(reduce_out)
            if i > 0 and i < self.backbone_end_level - 1:
                parallel_reduce_out = self.parallel_reduce_convs[i - 1](input)
                parallel_reduce_outs.append(parallel_reduce_out)

        # build top-down path
        # 插入 reduce_out[4, 5]
        inner_outs = [reduce_outs[-2], reduce_outs[-1]]
        # 3, 2, 1 更新 reduce_outs[0, 1, 2]
        for i in range(self.backbone_end_level - 1, 0, -1):
            feat_high_small = inner_outs[0]
            feat_low_large = reduce_outs[i - 1]
            upsample_feat = self.upsample_modules[i - 1](feat_high_small)
            inner_out = self.fpn_convs[i -
                                       1](torch.cat([upsample_feat, feat_low_large], 1))
            inner_outs.insert(0, inner_out)

        # build bottom-up path
        outs = [inner_outs[0]]
        for i in range(1, self.num_outs):
            feat_low_large = outs[-1]
            feat_high_small = inner_outs[i]
            downsample_feat = self.downsample_convs[i - 1](feat_low_large)
            if i == 1:
                feat_parallel = parallel_reduce_outs[i - 1]
                out = self.pafpn_convs[i - 1](torch.cat([downsample_feat, feat_parallel, feat_high_small], 1))
                outs.append(out)
            elif i == 2:
                feat_parallel = parallel_reduce_outs[i - 1]
                skip_feat_low_large = outs[-2]
                skip_downsample_feat = self.skip_downsample_convs[i - 2](
                    skip_feat_low_large)
                out = self.pafpn_convs[i - 1](torch.cat([skip_downsample_feat, downsample_feat, feat_parallel, feat_high_small], 1))
                outs.append(out)
            elif i == 3:
                skip_feat_low_large = outs[-2]
                skip_downsample_feat = self.skip_downsample_convs[i - 2](
                    skip_feat_low_large)
                out = self.pafpn_convs[i - 1](torch.cat([skip_downsample_feat, downsample_feat, feat_high_small], 1))
                outs.append(out)
            elif i == 4:
                if self.add_extra_convs == 'on_reduce':
                    skip_feat_low_large = outs[-2]
                    skip_downsample_feat = self.skip_downsample_convs[i - 2](
                        skip_feat_low_large)
                    out = self.pafpn_convs[i - 1](torch.cat([skip_downsample_feat, downsample_feat, feat_high_small], 1))
                    outs.append(out)   
                elif self.add_extra_convs == 'on_output':
                    skip_feat_low_large = outs[-2]
                    skip_downsample_feat = self.skip_downsample_convs[i - 2](
                        skip_feat_low_large)
                    out = self.pafpn_convs[i - 1](torch.cat([skip_downsample_feat, downsample_feat], 1))
                    outs.append(out)  
                elif self.add_extra_convs == 'on_reduce_wo_output':
                    skip_feat_low_large = outs[-2]
                    skip_downsample_feat = self.skip_downsample_convs[i - 2](
                        skip_feat_low_large)
                    out = self.pafpn_convs[i - 1](torch.cat([skip_downsample_feat, feat_high_small], 1))
                    outs.append(out)  
                else:
                    raise NotImplementedError(
                        f'SKIPPAFPN {self.add_extra_convs} extra_convs layer are not implemented !')
            else:
                raise NotImplementedError(
                    f'SKIPPAFPN {i + 1} layer are not implemented !')

        return tuple(outs)


@MODELS.register_module()
class PAFPN_CARAFE_Skip(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Defaults to 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Defaults to -1, which means the
            last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Defaults to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Defaults to False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Defaults to False.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Defaults to None.
        act_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            activation layer in ConvModule. Defaults to None.
        upsample_cfg (:obj:`ConfigDict` or dict, optional): Config dict
            for interpolate layer. Defaults to dict(mode='nearest').
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: Union[bool, str] = False,
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        upsample_cfg: ConfigType = dict(type='nearest'),
        init_cfg: MultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.upsample = self.upsample_cfg.get('type')

        assert self.upsample in [
            'nearest', 'bilinear', 'deconv', 'pixel_shuffle', 'carafe', None
        ]
        if self.upsample in ['deconv', 'pixel_shuffle']:
            assert hasattr(
                self.upsample_cfg,
                'upsample_kernel') and self.upsample_cfg.upsample_kernel > 0
            self.upsample_kernel = self.upsample_cfg.pop('upsample_kernel')

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level

        # reduce
        self.reduce_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            reduce_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            self.reduce_convs.append(reduce_conv)
        # upsample
        self.upsample_modules = ModuleList()
        for i in range(self.start_level, self.backbone_end_level - 1):
            upsample_cfg_ = self.upsample_cfg.copy()
            if self.upsample == 'deconv':
                upsample_cfg_.update(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=self.upsample_kernel,
                    stride=2,
                    padding=(self.upsample_kernel - 1) // 2,
                    output_padding=(self.upsample_kernel - 1) // 2)
            elif self.upsample == 'pixel_shuffle':
                upsample_cfg_.update(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    scale_factor=2,
                    upsample_kernel=self.upsample_kernel)
            elif self.upsample == 'carafe':
                upsample_cfg_.update(channels=out_channels, scale_factor=2)
                print(f'[USE CARAFE upsample op]')
            else:
                # suppress warnings
                align_corners = (None
                                 if self.upsample == 'nearest' else False)
                upsample_cfg_.update(
                    scale_factor=2,
                    mode=self.upsample,
                    align_corners=align_corners)
            upsample_module = build_upsample_layer(upsample_cfg_)
            self.upsample_modules.append(upsample_module)
        # top-down
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level - 1):
            fpn_conv = ConvModule(
                out_channels * 2,
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            self.fpn_convs.append(fpn_conv)
        # downsample_convs
        self.downsample_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level if self.num_outs > self.backbone_end_level else self.backbone_end_level - 1):
            downsample_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.downsample_convs.append(downsample_conv)
        self.downsample_Skip_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level - 1):
            downsample_Skip_conv = ConvModule(
                out_channels,
                out_channels,
                5,
                stride=4,
                padding=2,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.downsample_Skip_convs.append(downsample_Skip_conv)

        # bottom up
        self.pafpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level if self.num_outs > self.backbone_end_level else self.backbone_end_level - 1):
            pafpn_conv = ConvModule(
                out_channels * 2 if i == 0 or i == self.backbone_end_level - 1 else out_channels * 3,
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            self.pafpn_convs.append(pafpn_conv)

    def init_weights(self):
        """Initialize the weights of module."""
        super(PAFPN_CARAFE_Skip, self).init_weights()
        for m in self.modules():
            if isinstance(m, CARAFEPack):
                m.init_weights()

    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        assert len(inputs) == len(self.in_channels)

        # build reduce layer out
        reduce_outs = [
            reduce_conv(inputs[i + self.start_level])
            for i, reduce_conv in enumerate(self.reduce_convs)
        ]

        # build top-down path
        used_backbone_levels = len(reduce_outs)
        inner_outs = [reduce_outs[-1]]
        for i in range(used_backbone_levels - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[i - 1]
            upsample_feat = self.upsample_modules[i - 1](feat_high)
            inner_out = self.fpn_convs[i -
                                            1](torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(used_backbone_levels - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            if idx == 0:
                downsample_feat = self.downsample_convs[idx](feat_low)
                out = self.pafpn_convs[idx](
                    torch.cat([downsample_feat, feat_high], 1))
                outs.append(out)
            else:
                downsample_feat = self.downsample_convs[idx](feat_low)
                feat_low_skip = outs[-2]
                downsample_feat_skip = self.downsample_Skip_convs[idx - 1](
                    feat_low_skip)
                out = self.pafpn_convs[idx](
                    torch.cat([downsample_feat, downsample_feat_skip, feat_high], 1))
                outs.append(out)

        # add extra levels
        if self.num_outs > len(outs):
            assert self.num_outs == len(
                outs) + 1, 'only support p2 p3 p4 p5 + p6'
            feat_low = outs[-1]
            feat_low_skip = outs[-2]
            downsample_feat = self.downsample_convs[-1](feat_low)
            downsample_feat_skip = self.downsample_Skip_convs[-1](
                feat_low_skip)
            out = self.pafpn_convs[-1](
                torch.cat([downsample_feat, downsample_feat_skip], 1))
            outs.append(out)
        return tuple(outs)


@MODELS.register_module()
class PAFPN_CARAFE_Skip_Parallel_Old(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Defaults to 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Defaults to -1, which means the
            last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Defaults to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Defaults to False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Defaults to False.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Defaults to None.
        act_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            activation layer in ConvModule. Defaults to None.
        upsample_cfg (:obj:`ConfigDict` or dict, optional): Config dict
            for interpolate layer. Defaults to dict(mode='nearest').
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: Union[bool, str] = False,
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        upsample_cfg: ConfigType = dict(type='nearest'),
        init_cfg: MultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.upsample = self.upsample_cfg.get('type')

        assert self.upsample in [
            'nearest', 'bilinear', 'deconv', 'pixel_shuffle', 'carafe', None
        ]
        if self.upsample in ['deconv', 'pixel_shuffle']:
            assert hasattr(
                self.upsample_cfg,
                'upsample_kernel') and self.upsample_cfg.upsample_kernel > 0
            self.upsample_kernel = self.upsample_cfg.pop('upsample_kernel')

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level

        # reduce
        self.reduce_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            reduce_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            self.reduce_convs.append(reduce_conv)
        # parallel_reduce
        self.parallel_reduce_convs = nn.ModuleList()
        for i in range(self.start_level + 1, self.backbone_end_level - 1):
            parallel_reduce_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            self.parallel_reduce_convs.append(parallel_reduce_conv)
        # upsample
        self.upsample_modules = ModuleList()
        for i in range(self.start_level, self.backbone_end_level - 1):
            upsample_cfg_ = self.upsample_cfg.copy()
            if self.upsample == 'deconv':
                upsample_cfg_.update(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=self.upsample_kernel,
                    stride=2,
                    padding=(self.upsample_kernel - 1) // 2,
                    output_padding=(self.upsample_kernel - 1) // 2)
            elif self.upsample == 'pixel_shuffle':
                upsample_cfg_.update(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    scale_factor=2,
                    upsample_kernel=self.upsample_kernel)
            elif self.upsample == 'carafe':
                upsample_cfg_.update(channels=out_channels, scale_factor=2)
                print(f'[USE CARAFE upsample op]')
            else:
                # suppress warnings
                align_corners = (None
                                 if self.upsample == 'nearest' else False)
                upsample_cfg_.update(
                    scale_factor=2,
                    mode=self.upsample,
                    align_corners=align_corners)
            upsample_module = build_upsample_layer(upsample_cfg_)
            self.upsample_modules.append(upsample_module)
        # top-down
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level - 1):
            fpn_conv = ConvModule(
                out_channels * 2,
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            self.fpn_convs.append(fpn_conv)
        # downsample_convs
        self.downsample_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level if self.num_outs > self.backbone_end_level else self.backbone_end_level - 1):
            downsample_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.downsample_convs.append(downsample_conv)

        self.downsample_Skip_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level - 1):
            downsample_Skip_conv = ConvModule(
                out_channels,
                out_channels,
                5,
                stride=4,
                padding=2,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.downsample_Skip_convs.append(downsample_Skip_conv)

        # bottom up
        self.pafpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level if self.num_outs > self.backbone_end_level else self.backbone_end_level - 1):
            if i == 0:
                cur_in_channels = out_channels * 3
            elif i == 1:
                cur_in_channels = out_channels * 4
            elif i == 2:
                cur_in_channels = out_channels * 3
            elif i == 3:
                cur_in_channels = out_channels * 2
            pafpn_conv = ConvModule(
                cur_in_channels,
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            self.pafpn_convs.append(pafpn_conv)

    def init_weights(self):
        """Initialize the weights of module."""
        super(PAFPN_CARAFE_Skip_Parallel_Old, self).init_weights()
        for m in self.modules():
            if isinstance(m, CARAFEPack):
                m.init_weights()

    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        assert len(inputs) == len(self.in_channels)

        # build reduce layer out
        reduce_outs = [
            reduce_conv(inputs[i + self.start_level])
            for i, reduce_conv in enumerate(self.reduce_convs)
        ]
        parallel_reduce_outs = [
            parallel_reduce_conv(inputs[i + 1])
            for i, parallel_reduce_conv in enumerate(self.parallel_reduce_convs)
        ]
        # build top-down path
        used_backbone_levels = len(reduce_outs)
        inner_outs = [reduce_outs[-1]]
        for i in range(used_backbone_levels - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[i - 1]
            upsample_feat = self.upsample_modules[i - 1](feat_high)
            inner_out = self.fpn_convs[i -
                                            1](torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(used_backbone_levels - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            if idx == 0:
                feat_parallel = parallel_reduce_outs[0]
                out = self.pafpn_convs[idx](
                    torch.cat([downsample_feat, feat_parallel, feat_high], 1))
                outs.append(out)
            elif idx == 1:
                feat_low_skip = outs[-2]
                downsample_feat_skip = self.downsample_Skip_convs[idx - 1](
                    feat_low_skip)
                feat_parallel = parallel_reduce_outs[1]
                out = self.pafpn_convs[idx](
                    torch.cat([downsample_feat_skip, downsample_feat, feat_parallel, feat_high], 1))
                outs.append(out)
            elif idx == 2:
                feat_low_skip = outs[-2]
                downsample_feat_skip = self.downsample_Skip_convs[idx - 1](
                    feat_low_skip)
                out = self.pafpn_convs[idx](
                    torch.cat([downsample_feat_skip, downsample_feat, feat_high], 1))
                outs.append(out)

        # add extra levels
        if self.num_outs > len(outs):
            assert self.num_outs == len(
                outs) + 1, 'only support p2 p3 p4 p5 + p6'
            feat_low = outs[-1]
            feat_low_skip = outs[-2]
            downsample_feat = self.downsample_convs[-1](feat_low)
            downsample_feat_skip = self.downsample_Skip_convs[-1](
                feat_low_skip)
            out = self.pafpn_convs[-1](
                torch.cat([downsample_feat, downsample_feat_skip], 1))
            outs.append(out)
        return tuple(outs)


@MODELS.register_module()
class ImprovedPAFPN(BaseModule):
    def __init__(
        self,
        use_type: str,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        concat_kernel_size: int = 1,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: Union[bool, str] = 'on_reduce',
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        upsample_cfg: ConfigType = dict(type='nearest'),
        init_cfg: MultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super(ImprovedPAFPN, self).__init__(None)
        if use_type == 'PAFPN_CARAFE_Skip_Parallel':
            self.improvedpafpn = PAFPN_CARAFE_Skip_Parallel(
                in_channels=in_channels,
                out_channels=out_channels,
                num_outs=num_outs,
                add_extra_convs=add_extra_convs,
                start_level=start_level,
                end_level=end_level,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                upsample_cfg=upsample_cfg
            )
        elif use_type == 'PAFPN_CARAFE_Skip_Parallel_concat':
            self.improvedpafpn = PAFPN_CARAFE_Skip_Parallel_concat(
                in_channels=in_channels,
                out_channels=out_channels,
                num_outs=num_outs,
                concat_kernel_size=concat_kernel_size,
                add_extra_convs=add_extra_convs,
                start_level=start_level,
                end_level=end_level,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                upsample_cfg=upsample_cfg
            )
        elif use_type == 'PAFPN_CARAFE_Skip':
            self.improvedpafpn = PAFPN_CARAFE_Skip(
                in_channels=in_channels,
                out_channels=out_channels,
                num_outs=num_outs,
                start_level=start_level,
                end_level=end_level,
                add_extra_convs=add_extra_convs,
                relu_before_extra_convs=relu_before_extra_convs,
                no_norm_on_lateral=no_norm_on_lateral,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                upsample_cfg=upsample_cfg,
                init_cfg=init_cfg
            )
        elif use_type == 'PAFPN_CARAFE_Skip_Parallel_Old':
            self.improvedpafpn = PAFPN_CARAFE_Skip_Parallel_Old(
                in_channels=in_channels,
                out_channels=out_channels,
                num_outs=num_outs,
                start_level=start_level,
                end_level=end_level,
                add_extra_convs=add_extra_convs,
                relu_before_extra_convs=relu_before_extra_convs,
                no_norm_on_lateral=no_norm_on_lateral,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                upsample_cfg=upsample_cfg,
                init_cfg=init_cfg
            )
        else:
            raise NotImplementedError(f'The {use_type} are not implemented !')

    def forward(self, x):
        x = self.improvedpafpn(x)
        return x