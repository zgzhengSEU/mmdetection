# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import kaiming_init
from mmengine.registry import MODELS

def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction = 16, freq_sel_method = 'top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x] 
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,h,w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered. 
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2,3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                        
        return dct_filter

@MODELS.register_module()
class TransformerWithFCA(nn.Module):
    _abbr_ = 'gen_attention_block'

    def __init__(self,
                 in_channels: int,
                 num_heads: int = 8, # 8
                 kv_stride: int = 2,
                 reduction: int = 16,
                 FCA_pos: str = 'before',
                 pos_after_conv: int = 2):

        super().__init__()
        c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
        if pos_after_conv == 2:
            planes = in_channels
            reduction = reduction // 4
        else:
            planes = in_channels // 4
        self.channel_att = MultiSpectralAttentionLayer(in_channels, c2wh[planes], c2wh[planes],  reduction=reduction, freq_sel_method = 'top16')
        # hard range means local range for non-local operation
        self.FCA_pos = FCA_pos
        # 8
        self.num_heads = num_heads
        self.in_channels = in_channels
        # -1
        # 2
        self.kv_stride = kv_stride
        # 1
        self.qk_embed_dim = in_channels // num_heads
        out_c = self.qk_embed_dim * num_heads

        self.key_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_c,
            kernel_size=1,
            bias=False)
        self.key_conv.kaiming_init = True

        self.v_dim = in_channels // num_heads
        self.value_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.v_dim * num_heads,
            kernel_size=1,
            bias=False)
        self.value_conv.kaiming_init = True


        stdv = 1.0 / math.sqrt(self.qk_embed_dim * 2)
        appr_bias_value = -2 * stdv * torch.rand(out_c) + stdv
        # in_channel维可训练参数 128
        self.appr_bias = nn.Parameter(appr_bias_value)


        self.proj_conv = nn.Conv2d(
            in_channels=self.v_dim * num_heads,
            out_channels=in_channels,
            kernel_size=1,
            bias=True)
        self.proj_conv.kaiming_init = True
        self.gamma = nn.Parameter(torch.zeros(1))


        self.kv_downsample = nn.AvgPool2d(
                kernel_size=1, stride=self.kv_stride)

        self.init_weights()

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        if self.FCA_pos == 'before_TR':
            x_input = self.channel_att(x_input)
        
        num_heads = self.num_heads # 8

        n, _, h, w = x_input.shape
        
        if self.FCA_pos == 'before_kv':
            x_kv = self.kv_downsample(self.channel_att(x_input))
        else:
            x_kv = self.kv_downsample(x_input)
            
        _, _, h_kv, w_kv = x_kv.shape


        proj_key = self.key_conv(x_kv).view(
            (n, num_heads, self.qk_embed_dim, h_kv * w_kv))

        # accelerate for saliency only
        appr_bias = self.appr_bias.\
            view(1, num_heads, 1, self.qk_embed_dim).\
            repeat(n, 1, h_kv * w_kv, 1)
        print(proj_key.shape)
        print(appr_bias.shape)
        energy = torch.matmul(appr_bias, proj_key).\
            view(n, num_heads, h_kv * w_kv, h_kv * w_kv)

        # h = 1
        # w = 1
        # h = h_kv
        # w = w_kv
        attention = F.softmax(energy, 3)

        proj_value = self.value_conv(x_kv)
        
        if self.FCA_pos == 'after_value':
            proj_value = self.channel_att(proj_value)
        
        proj_value_reshape = proj_value.\
            view((n, num_heads, self.v_dim, h_kv * w_kv)).\
            permute(0, 1, 3, 2)

        out = torch.matmul(attention, proj_value_reshape).\
            permute(0, 1, 3, 2).\
            contiguous().\
            view(n, self.v_dim * self.num_heads, h_kv, w_kv)

        out = self.proj_conv(out)

        if self.FCA_pos == 'parallel_TR_Noshortcut':
            out = self.gamma * out + self.channel_att(x_input)
        else:
            out = self.gamma * nn.functional.interpolate(out, (h, w))  + x_input
        
        if self.FCA_pos == 'parallel_TR':
            out += self.channel_att(x_input)
        
        if self.FCA_pos == 'after_TR':
            out = self.channel_att(out)
            
        return out

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'kaiming_init') and m.kaiming_init:
                kaiming_init(
                    m,
                    mode='fan_in',
                    nonlinearity='leaky_relu',
                    bias=0,
                    distribution='uniform',
                    a=1)

if __name__ == '__main__':
    
    # # test attention_type='1000'
    # imgs = torch.randn(2, 16, 20, 20)
    # gen_attention_block = GeneralizedAttention(16, attention_type='1000')
    # assert gen_attention_block.query_conv.in_channels == 16
    # assert gen_attention_block.key_conv.in_channels == 16
    # assert gen_attention_block.key_conv.in_channels == 16
    # out = gen_attention_block(imgs)
    # assert out.shape == imgs.shape

    # # test attention_type='0100'
    # imgs = torch.randn(2, 16, 20, 20)
    # gen_attention_block = GeneralizedAttention(16, attention_type='0100')
    # assert gen_attention_block.query_conv.in_channels == 16
    # assert gen_attention_block.appr_geom_fc_x.in_features == 8
    # assert gen_attention_block.appr_geom_fc_y.in_features == 8
    # out = gen_attention_block(imgs)
    # assert out.shape == imgs.shape

    # test attention_type='0010'
    imgs = torch.randn(8, 128, 20, 20)
    gen_attention_block = TransformerWithFCA(in_channels=128, num_heads=8)
    assert gen_attention_block.key_conv.in_channels == 128
    assert hasattr(gen_attention_block, 'appr_bias')
    out = gen_attention_block(imgs)
    print(out.shape)
    assert out.shape == imgs.shape

    # # test attention_type='0001'
    # imgs = torch.randn(2, 16, 20, 20)
    # gen_attention_block = GeneralizedAttention(16, attention_type='0001')
    # assert gen_attention_block.appr_geom_fc_x.in_features == 8
    # assert gen_attention_block.appr_geom_fc_y.in_features == 8
    # assert hasattr(gen_attention_block, 'geom_bias')
    # out = gen_attention_block(imgs)
    # assert out.shape == imgs.shape

    # # test spatial_range >= 0
    # imgs = torch.randn(2, 256, 20, 20)
    # gen_attention_block = GeneralizedAttention(256, spatial_range=10)
    # assert hasattr(gen_attention_block, 'local_constraint_map')
    # out = gen_attention_block(imgs)
    # assert out.shape == imgs.shape

    # # test q_stride > 1
    # imgs = torch.randn(2, 16, 20, 20)
    # gen_attention_block = GeneralizedAttention(16, q_stride=2)
    # assert gen_attention_block.q_downsample is not None
    # out = gen_attention_block(imgs)
    # assert out.shape == imgs.shape

    # # test kv_stride > 1
    # imgs = torch.randn(2, 16, 20, 20)
    # gen_attention_block = GeneralizedAttention(16, kv_stride=2)
    # assert gen_attention_block.kv_downsample is not None
    # out = gen_attention_block(imgs)
    # assert out.shape == imgs.shape

    # # test fp16 with attention_type='1111'
    # if torch.cuda.is_available():
    #     imgs = torch.randn(2, 16, 20, 20).cuda().to(torch.half)
    #     gen_attention_block = GeneralizedAttention(
    #         16,
    #         spatial_range=-1,
    #         num_heads=8,
    #         attention_type='1111',
    #         kv_stride=2)
    #     gen_attention_block.cuda().type(torch.half)
    #     out = gen_attention_block(imgs)
    #     assert out.shape == imgs.shape