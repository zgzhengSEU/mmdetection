# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, ModuleList
from torch import Tensor

# from mmdet.models.backbones.resnet import Bottleneck
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptConfigType, OptMultiConfig
from .bbox_head import BBoxHead

# new add
from ...layers import CoordConv2d
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
import torch.utils.checkpoint as cp

class BasicResBlock(BaseModule):
    """Basic residual block.

    This block is a little different from the block in the ResNet backbone.
    The kernel size of conv1 is 1 in this block while 3 in ResNet BasicBlock.

    Args:
        in_channels (int): Channels of the input feature map.
        out_channels (int): Channels of the output feature map.
        conv_cfg (:obj:`ConfigDict` or dict, optional): The config dict
            for convolution layers.
        norm_cfg (:obj:`ConfigDict` or dict): The config dict for
            normalization layers.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 init_cfg: OptMultiConfig = None,
                 use_coordconv: bool = False) -> None:
        super().__init__(init_cfg=init_cfg)

        # main path
        if use_coordconv: 
            self.conv1 = CoordConv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                with_r=True)
        else:
            self.conv1 = ConvModule(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
        self.conv2 = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        # identity path
        self.conv_identity = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)

        identity = self.conv_identity(identity)
        out = x + identity

        out = self.relu(out)
        return out


class Bottleneck(BaseModule):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 plugins=None,
                 init_cfg=None,
                 use_coordconv=False):
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(init_cfg)
        assert style in ['pytorch', 'caffe']
        assert plugins is None or isinstance(plugins, list)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.use_coordconv = use_coordconv
        
        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)

        if self.use_coordconv:
            self.conv2 = CoordConv2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False,
                conv_cfg=conv_cfg,
                with_r=True
            )
        else:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        self.add_module(self.norm2_name, norm2)
        
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)


            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)


            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        out = _inner_forward(x)

        out = self.relu(out)

        return out


@MODELS.register_module()
class DoubleConvFCBBoxHead(BBoxHead):
    r"""Bbox head used in Double-Head R-CNN

    .. code-block:: none

                                          /-> cls
                      /-> shared convs ->
                                          \-> reg
        roi features
                                          /-> cls
                      \-> shared fc    ->
                                          \-> reg
    """  # noqa: W605

    def __init__(self,
                 use_coordconv: bool = False,
                 num_convs: int = 0,
                 num_fcs: int = 0,
                 conv_out_channels: int = 1024,
                 fc_out_channels: int = 1024,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 init_cfg: MultiConfig = dict(
                     type='Normal',
                     override=[
                         dict(type='Normal', name='fc_cls', std=0.01),
                         dict(type='Normal', name='fc_reg', std=0.001),
                         dict(
                             type='Xavier',
                             name='fc_branch',
                             distribution='uniform')
                     ]),
                 **kwargs) -> None:
        kwargs.setdefault('with_avg_pool', True)
        super().__init__(init_cfg=init_cfg, **kwargs)
        assert self.with_avg_pool
        assert num_convs > 0
        assert num_fcs > 0
        self.num_convs = num_convs
        self.num_fcs = num_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.use_coordconv = use_coordconv
        # increase the channel of input features
        self.res_block = BasicResBlock(self.in_channels,
                                       self.conv_out_channels)

        # add conv heads
        self.conv_branch = self._add_conv_branch()
        # add fc heads
        self.fc_branch = self._add_fc_branch()

        out_dim_reg = 4 if self.reg_class_agnostic else 4 * self.num_classes
        self.fc_reg = nn.Linear(self.conv_out_channels, out_dim_reg)

        self.fc_cls = nn.Linear(self.fc_out_channels, self.num_classes + 1)
        self.relu = nn.ReLU()

    def _add_conv_branch(self) -> None:
        """Add the fc branch which consists of a sequential of conv layers."""
        branch_convs = ModuleList()
        for i in range(self.num_convs):
            branch_convs.append(
                Bottleneck(
                    inplanes=self.conv_out_channels,
                    planes=self.conv_out_channels // 4,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        return branch_convs

    def _add_fc_branch(self) -> None:
        """Add the fc branch which consists of a sequential of fc layers."""
        branch_fcs = ModuleList()
        for i in range(self.num_fcs):
            fc_in_channels = (
                self.in_channels *
                self.roi_feat_area if i == 0 else self.fc_out_channels)
            branch_fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
        return branch_fcs

    def forward(self, x_cls: Tensor, x_reg: Tensor) -> Tuple[Tensor]:
        """Forward features from the upstream network.

        Args:
            x_cls (Tensor): Classification features of rois
            x_reg (Tensor): Regression features from the upstream network.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_score (Tensor): Classification score predictions of rois.
                  each roi predicts num_classes + 1 channels.
                - bbox_pred (Tensor): BBox deltas predictions of rois. each roi
                  predicts 4 * num_classes channels.
        """
        # conv head
        x_conv = self.res_block(x_reg)

        for conv in self.conv_branch:
            x_conv = conv(x_conv)

        if self.with_avg_pool:
            x_conv = self.avg_pool(x_conv)

        x_conv = x_conv.view(x_conv.size(0), -1)
        bbox_pred = self.fc_reg(x_conv)

        # fc head
        x_fc = x_cls.view(x_cls.size(0), -1)
        for fc in self.fc_branch:
            x_fc = self.relu(fc(x_fc))

        cls_score = self.fc_cls(x_fc)

        return cls_score, bbox_pred
