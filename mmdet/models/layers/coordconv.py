import torch
import torch.nn as nn
import torch.nn.modules.conv as conv
from typing import Dict, Optional, Tuple, Union
from mmcv.cnn import ConvModule

class AddCoords(nn.Module):
    def __init__(self, with_r=False, use_cuda=True):
        super(AddCoords, self).__init__()
        self.with_r = with_r
        self.use_cuda = use_cuda

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """

        batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
        xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
        yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

        xx_range = torch.arange(dim_y, dtype=torch.int32)
        yy_range = torch.arange(dim_x, dtype=torch.int32)
        xx_range = xx_range[None, None, :, None]
        yy_range = yy_range[None, None, :, None]

        xx_channel = torch.matmul(xx_range, xx_ones)
        yy_channel = torch.matmul(yy_range, yy_ones)

        # transpose y
        yy_channel = yy_channel.permute(0, 1, 3, 2)

        xx_channel = xx_channel.float() / (dim_y - 1)
        yy_channel = yy_channel.float() / (dim_x - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

        if torch.cuda.is_available and self.use_cuda:
            input_tensor = input_tensor.cuda()
            xx_channel = xx_channel.cuda()
            yy_channel = yy_channel.cuda()
        out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            out = torch.cat([out, rr], dim=1)

        return out

class CoordConv2d(conv.Conv2d):
    _abbr_ = 'coord_conv_block'
    def __init__(self,
                in_channels: int,
                out_channels: int,
                kernel_size: Union[int, Tuple[int, int]],
                stride: Union[int, Tuple[int, int]] = 1,
                padding: Union[int, Tuple[int, int]] = 0,
                dilation: Union[int, Tuple[int, int]] = 1,
                groups: int = 1,
                bias: Union[bool, str] = 'auto',
                conv_cfg: Optional[Dict] = None,
                norm_cfg: Optional[Dict] = None,
                act_cfg: Optional[Dict] = dict(type='ReLU'),
                inplace: bool = True,
                with_spectral_norm: bool = False,
                padding_mode: str = 'zeros',
                order: tuple = ('conv', 'norm', 'act'),
                with_r=False,
                use_cuda=True):

        super().__init__()
        self.addcoords = AddCoords(with_r, use_cuda=use_cuda)
        self.conv = ConvModule(
            in_channels + 2 + int(with_r), 
            out_channels,
            kernel_size, 
            stride, 
            padding, 
            dilation, 
            groups, 
            bias, 
            conv_cfg,
            norm_cfg,
            act_cfg,
            inplace,
            with_spectral_norm,
            padding_mode,
            order)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in, H, W)
        output_tensor_shape: (N, C_out, H_out, W_out)
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out
