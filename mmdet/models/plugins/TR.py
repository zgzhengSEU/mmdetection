# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import kaiming_init
from mmengine.registry import MODELS
from mmcv.cnn import ConvModule, Scale

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
                 para_spatial: bool = False,
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
        self.para_spatial = para_spatial
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
        if self.para_spatial:
            appr_bias = self.appr_bias.\
                view(1, num_heads, 1, self.qk_embed_dim).\
                repeat(n, 1, h_kv * w_kv, 1)
        else:
            appr_bias = self.appr_bias.\
                view(1, num_heads, 1, self.qk_embed_dim).\
                repeat(n, 1, 1, 1)
        # print(proj_key.shape)
        # print(appr_bias.shape)
        energy = torch.matmul(appr_bias, proj_key).\
            view(n, num_heads, h_kv * w_kv if self.para_spatial else 1, h_kv * w_kv)

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
            view(n, self.v_dim * self.num_heads, h_kv if self.para_spatial else 1, w_kv if self.para_spatial else 1)

        out = self.proj_conv(out)

        if self.FCA_pos == 'parallel_TR_Noshortcut':
            out = self.gamma * out + self.channel_att(x_input)
        else:
            if self.para_spatial:
                out = self.gamma * nn.functional.interpolate(out, (h, w))  + x_input
            else:
                out = self.gamma * out + x_input
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

@MODELS.register_module()
class TR(nn.Module):
    """GeneralizedAttention module.

    See 'An Empirical Study of Spatial Attention Mechanisms in Deep Networks'
    (https://arxiv.org/abs/1904.05873) for details.

    Args:
        in_channels (int): Channels of the input feature map.
        spatial_range (int): The spatial range. -1 indicates no spatial range
            constraint. Default: -1.
        num_heads (int): The head number of empirical_attention module.
            Default: 9.
        position_embedding_dim (int): The position embedding dimension.
            Default: -1.
        position_magnitude (int): A multiplier acting on coord difference.
            Default: 1.
        kv_stride (int): The feature stride acting on key/value feature map.
            Default: 2.
        q_stride (int): The feature stride acting on query feature map.
            Default: 1.
        attention_type (str): A binary indicator string for indicating which
            items in generalized empirical_attention module are used.
            Default: '1111'.

            - '1000' indicates 'query and key content' (appr - appr) item,
            - '0100' indicates 'query content and relative position'
              (appr - position) item,
            - '0010' indicates 'key content only' (bias - appr) item,
            - '0001' indicates 'relative position only' (bias - position) item.
    """

    _abbr_ = 'gen_attention_block'

    def __init__(self,
                 in_channels: int,
                 spatial_range: int = -1,
                 num_heads: int = 8,
                 position_embedding_dim: int = -1,
                 position_magnitude: int = 1,
                 kv_stride: int = 2,
                 q_stride: int = 2,
                 reduction: int = 16,
                 FCA_pos: str = 'none',
                 ResNext: bool = False,
                 pos_after_conv: int=2,
                 attention_type: str = '0110'):

        super().__init__()

        c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
        if pos_after_conv == 2:
            planes = in_channels // 2 if ResNext else in_channels
            reduction = reduction // 4
        else:
            planes = in_channels // 2 if ResNext else in_channels // 4
        self.channel_att = MultiSpectralAttentionLayer(in_channels, c2wh[planes], c2wh[planes],  reduction=reduction, freq_sel_method = 'top16')
        self.FCA_pos = FCA_pos
        # hard range means local range for non-local operation
        self.position_embedding_dim = (
            position_embedding_dim
            if position_embedding_dim > 0 else in_channels)

        self.position_magnitude = position_magnitude
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.spatial_range = spatial_range
        self.kv_stride = kv_stride
        self.q_stride = q_stride
        self.attention_type = [bool(int(_)) for _ in attention_type]
        self.qk_embed_dim = in_channels // num_heads
        out_c = self.qk_embed_dim * num_heads

        if self.attention_type[0] or self.attention_type[1]:
            self.query_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_c,
                kernel_size=1,
                bias=False)
            self.query_conv.kaiming_init = True

        if self.attention_type[0] or self.attention_type[2]:
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

        if self.attention_type[1] or self.attention_type[3]:
            self.appr_geom_fc_x = nn.Linear(
                self.position_embedding_dim // 2, out_c, bias=False)
            self.appr_geom_fc_x.kaiming_init = True

            self.appr_geom_fc_y = nn.Linear(
                self.position_embedding_dim // 2, out_c, bias=False)
            self.appr_geom_fc_y.kaiming_init = True

        if self.attention_type[2]:
            stdv = 1.0 / math.sqrt(self.qk_embed_dim * 2)
            appr_bias_value = -2 * stdv * torch.rand(out_c) + stdv
            self.appr_bias = nn.Parameter(appr_bias_value)

        if self.attention_type[3]:
            stdv = 1.0 / math.sqrt(self.qk_embed_dim * 2)
            geom_bias_value = -2 * stdv * torch.rand(out_c) + stdv
            self.geom_bias = nn.Parameter(geom_bias_value)

        self.proj_conv = nn.Conv2d(
            in_channels=self.v_dim * num_heads,
            out_channels=in_channels,
            kernel_size=1,
            bias=True)
        self.proj_conv.kaiming_init = True
        self.gamma = nn.Parameter(torch.zeros(1))

        if self.spatial_range >= 0:
            # only works when non local is after 3*3 conv
            if in_channels == 256:
                max_len = 84
            elif in_channels == 512:
                max_len = 42

            max_len_kv = int((max_len - 1.0) / self.kv_stride + 1)
            local_constraint_map = np.ones(
                (max_len, max_len, max_len_kv, max_len_kv), dtype=int)
            for iy in range(max_len):
                for ix in range(max_len):
                    local_constraint_map[
                        iy, ix,
                        max((iy - self.spatial_range) //
                            self.kv_stride, 0):min((iy + self.spatial_range +
                                                    1) // self.kv_stride +
                                                   1, max_len),
                        max((ix - self.spatial_range) //
                            self.kv_stride, 0):min((ix + self.spatial_range +
                                                    1) // self.kv_stride +
                                                   1, max_len)] = 0

            self.local_constraint_map = nn.Parameter(
                torch.from_numpy(local_constraint_map).byte(),
                requires_grad=False)

        if self.q_stride > 1:
            self.q_downsample = nn.AvgPool2d(
                kernel_size=1, stride=self.q_stride)
        else:
            self.q_downsample = None

        if self.kv_stride > 1:
            self.kv_downsample = nn.AvgPool2d(
                kernel_size=1, stride=self.kv_stride)
        else:
            self.kv_downsample = None

        self.init_weights()

    def get_position_embedding(self,
                               h,
                               w,
                               h_kv,
                               w_kv,
                               q_stride,
                               kv_stride,
                               device,
                               dtype,
                               feat_dim,
                               wave_length=1000):
        # the default type of Tensor is float32, leading to type mismatch
        # in fp16 mode. Cast it to support fp16 mode.
        h_idxs = torch.linspace(0, h - 1, h).to(device=device, dtype=dtype)
        h_idxs = h_idxs.view((h, 1)) * q_stride

        w_idxs = torch.linspace(0, w - 1, w).to(device=device, dtype=dtype)
        w_idxs = w_idxs.view((w, 1)) * q_stride

        h_kv_idxs = torch.linspace(0, h_kv - 1, h_kv).to(
            device=device, dtype=dtype)
        h_kv_idxs = h_kv_idxs.view((h_kv, 1)) * kv_stride

        w_kv_idxs = torch.linspace(0, w_kv - 1, w_kv).to(
            device=device, dtype=dtype)
        w_kv_idxs = w_kv_idxs.view((w_kv, 1)) * kv_stride

        # (h, h_kv, 1)
        h_diff = h_idxs.unsqueeze(1) - h_kv_idxs.unsqueeze(0)
        h_diff *= self.position_magnitude

        # (w, w_kv, 1)
        w_diff = w_idxs.unsqueeze(1) - w_kv_idxs.unsqueeze(0)
        w_diff *= self.position_magnitude

        feat_range = torch.arange(0, feat_dim / 4).to(
            device=device, dtype=dtype)

        dim_mat = torch.Tensor([wave_length]).to(device=device, dtype=dtype)
        dim_mat = dim_mat**((4. / feat_dim) * feat_range)
        dim_mat = dim_mat.view((1, 1, -1))

        embedding_x = torch.cat(
            ((w_diff / dim_mat).sin(), (w_diff / dim_mat).cos()), dim=2)

        embedding_y = torch.cat(
            ((h_diff / dim_mat).sin(), (h_diff / dim_mat).cos()), dim=2)

        return embedding_x, embedding_y

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        if self.FCA_pos == 'before_TR':
            x_input = self.channel_att(x_input)
        
        num_heads = self.num_heads

        # use empirical_attention
        if self.q_downsample is not None:
            x_q = self.q_downsample(x_input)
        else:
            x_q = x_input
        n, _, h, w = x_q.shape

        if self.kv_downsample is not None:
            x_kv = self.kv_downsample(x_input)
        else:
            x_kv = x_input
        _, _, h_kv, w_kv = x_kv.shape

        if self.attention_type[0] or self.attention_type[1]:
            proj_query = self.query_conv(x_q).view(
                (n, num_heads, self.qk_embed_dim, h * w))
            proj_query = proj_query.permute(0, 1, 3, 2)

        if self.attention_type[0] or self.attention_type[2]:
            proj_key = self.key_conv(x_kv).view(
                (n, num_heads, self.qk_embed_dim, h_kv * w_kv))

        if self.attention_type[1] or self.attention_type[3]:
            position_embed_x, position_embed_y = self.get_position_embedding(
                h, w, h_kv, w_kv, self.q_stride, self.kv_stride,
                x_input.device, x_input.dtype, self.position_embedding_dim)
            # (n, num_heads, w, w_kv, dim)
            position_feat_x = self.appr_geom_fc_x(position_embed_x).\
                view(1, w, w_kv, num_heads, self.qk_embed_dim).\
                permute(0, 3, 1, 2, 4).\
                repeat(n, 1, 1, 1, 1)

            # (n, num_heads, h, h_kv, dim)
            position_feat_y = self.appr_geom_fc_y(position_embed_y).\
                view(1, h, h_kv, num_heads, self.qk_embed_dim).\
                permute(0, 3, 1, 2, 4).\
                repeat(n, 1, 1, 1, 1)

            position_feat_x /= math.sqrt(2)
            position_feat_y /= math.sqrt(2)

        # accelerate for saliency only
        if (np.sum(self.attention_type) == 1) and self.attention_type[2]:
            appr_bias = self.appr_bias.\
                view(1, num_heads, 1, self.qk_embed_dim).\
                repeat(n, 1, 1, 1)

            energy = torch.matmul(appr_bias, proj_key).\
                view(n, num_heads, 1, h_kv * w_kv)

            h = 1
            w = 1
        else:
            # (n, num_heads, h*w, h_kv*w_kv), query before key, 540mb for
            if not self.attention_type[0]:
                energy = torch.zeros(
                    n,
                    num_heads,
                    h,
                    w,
                    h_kv,
                    w_kv,
                    dtype=x_input.dtype,
                    device=x_input.device)

            # attention_type[0]: appr - appr
            # attention_type[1]: appr - position
            # attention_type[2]: bias - appr
            # attention_type[3]: bias - position
            if self.attention_type[0] or self.attention_type[2]:
                if self.attention_type[0] and self.attention_type[2]:
                    appr_bias = self.appr_bias.\
                        view(1, num_heads, 1, self.qk_embed_dim)
                    energy = torch.matmul(proj_query + appr_bias, proj_key).\
                        view(n, num_heads, h, w, h_kv, w_kv)

                elif self.attention_type[0]:
                    energy = torch.matmul(proj_query, proj_key).\
                        view(n, num_heads, h, w, h_kv, w_kv)

                elif self.attention_type[2]:
                    appr_bias = self.appr_bias.\
                        view(1, num_heads, 1, self.qk_embed_dim).\
                        repeat(n, 1, 1, 1)

                    energy += torch.matmul(appr_bias, proj_key).\
                        view(n, num_heads, 1, 1, h_kv, w_kv)

            if self.attention_type[1] or self.attention_type[3]:
                if self.attention_type[1] and self.attention_type[3]:
                    geom_bias = self.geom_bias.\
                        view(1, num_heads, 1, self.qk_embed_dim)

                    proj_query_reshape = (proj_query + geom_bias).\
                        view(n, num_heads, h, w, self.qk_embed_dim)

                    energy_x = torch.matmul(
                        proj_query_reshape.permute(0, 1, 3, 2, 4),
                        position_feat_x.permute(0, 1, 2, 4, 3))
                    energy_x = energy_x.\
                        permute(0, 1, 3, 2, 4).unsqueeze(4)

                    energy_y = torch.matmul(
                        proj_query_reshape,
                        position_feat_y.permute(0, 1, 2, 4, 3))
                    energy_y = energy_y.unsqueeze(5)

                    energy += energy_x + energy_y

                elif self.attention_type[1]:
                    proj_query_reshape = proj_query.\
                        view(n, num_heads, h, w, self.qk_embed_dim)
                    proj_query_reshape = proj_query_reshape.\
                        permute(0, 1, 3, 2, 4)
                    position_feat_x_reshape = position_feat_x.\
                        permute(0, 1, 2, 4, 3)
                    position_feat_y_reshape = position_feat_y.\
                        permute(0, 1, 2, 4, 3)

                    energy_x = torch.matmul(proj_query_reshape,
                                            position_feat_x_reshape)
                    energy_x = energy_x.permute(0, 1, 3, 2, 4).unsqueeze(4)
                    
                    proj_query_reshape = proj_query_reshape.permute(0, 1, 3, 2, 4)
                    energy_y = torch.matmul(proj_query_reshape,
                                            position_feat_y_reshape)
                    energy_y = energy_y.unsqueeze(5)

                    energy += energy_x + energy_y

                elif self.attention_type[3]:
                    geom_bias = self.geom_bias.\
                        view(1, num_heads, self.qk_embed_dim, 1).\
                        repeat(n, 1, 1, 1)

                    position_feat_x_reshape = position_feat_x.\
                        view(n, num_heads, w * w_kv, self.qk_embed_dim)

                    position_feat_y_reshape = position_feat_y.\
                        view(n, num_heads, h * h_kv, self.qk_embed_dim)

                    energy_x = torch.matmul(position_feat_x_reshape, geom_bias)
                    energy_x = energy_x.view(n, num_heads, 1, w, 1, w_kv)

                    energy_y = torch.matmul(position_feat_y_reshape, geom_bias)
                    energy_y = energy_y.view(n, num_heads, h, 1, h_kv, 1)

                    energy += energy_x + energy_y

            energy = energy.view(n, num_heads, h * w, h_kv * w_kv)

        if self.spatial_range >= 0:
            cur_local_constraint_map = \
                self.local_constraint_map[:h, :w, :h_kv, :w_kv].\
                contiguous().\
                view(1, 1, h*w, h_kv*w_kv)

            energy = energy.masked_fill_(cur_local_constraint_map,
                                         float('-inf'))

        attention = F.softmax(energy, 3)

        proj_value = self.value_conv(x_kv)
        proj_value_reshape = proj_value.\
            view((n, num_heads, self.v_dim, h_kv * w_kv)).\
            permute(0, 1, 3, 2)

        out = torch.matmul(attention, proj_value_reshape).\
            permute(0, 1, 3, 2).\
            contiguous().\
            view(n, self.v_dim * self.num_heads, h, w)

        out = self.proj_conv(out)

        # output is downsampled, upsample back to input size
        if self.q_downsample is not None:
            out = F.interpolate(
                out,
                size=x_input.shape[2:],
                mode='bilinear',
                align_corners=False)
            
        if self.FCA_pos == 'parallel_TR_Noshortcut':
            out = self.gamma * out + self.channel_att(x_input)
        else:    
            out = self.gamma * out + x_input
            
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

class CAM(nn.Module):
    """Channel Attention Module (CAM)"""

    def __init__(self):
        super(CAM, self).__init__()
        self.gamma = Scale(0)

    def forward(self, x):
        """Forward function."""
        batch_size, channels, height, width = x.size()
        proj_query = x.view(batch_size, channels, -1) # [8, 128, 800]
        proj_key = x.view(batch_size, channels, -1).permute(0, 2, 1) # [8, 800, 128]
        energy = torch.bmm(proj_query, proj_key) # [8, 128, 128]
        energy_new = torch.max(energy, -1, keepdim=True)[0] # [8, 128, 1]
        energy_new = energy_new.expand_as(energy) # [8, 128, 128]
        energy_new = energy_new - energy # [8, 128, 128]
        attention = F.softmax(energy_new, dim=-1) # [8, 128, 128]
        proj_value = x.view(batch_size, channels, -1) # [8, 128, 800]

        out = torch.bmm(attention, proj_value) # [8, 128, 800]
        out = out.view(batch_size, channels, height, width) # [8, 128, 40, 20]

        out = self.gamma(out) + x # [8, 128, 40, 20]
        return out


class CTR(nn.Module):
    """Channel Attention Module (CAM)"""

    def __init__(self, num_head):
        super(CTR, self).__init__()
        self.gamma = Scale(0)
        self.num_head = num_head

    def forward(self, x):
        """Forward function."""
        batch_size, channels, height, width = x.size()
        N = height * width
        dim = N // self.num_head
        proj_query = x.view(batch_size, channels, N)
        proj_query = proj_query.permute(0, 2, 1)
        proj_query = proj_query.contiguous().view(batch_size, self.num_head, dim, channels)
        proj_query = proj_query.permute(0, 1, 3, 2)
        
        proj_key = x.view(batch_size, channels, N).permute(0, 2, 1)
        proj_key = proj_key.contiguous().view(batch_size, self.num_head, dim, channels)
        
        energy = torch.matmul(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = F.softmax(energy_new, dim=-1)
        proj_value = x.view(batch_size, channels, N)
        proj_value = proj_value.permute(0, 2, 1)
        proj_value = proj_value.contiguous().view(batch_size, self.num_head, dim, channels)
        proj_value = proj_value.permute(0, 1, 3, 2)
        
        out = torch.matmul(attention, proj_value)
        out = out.permute(0, 1, 3, 2)
        out = out.contiguous().view(batch_size, self.num_head * dim, channels).permute(0, 2, 1)
        out = out.view(batch_size, channels, height, width)

        out = self.gamma(out) + x
        return out


@MODELS.register_module()
class ChannelTR(nn.Module):
    _abbr_ = 'ChannelTR_block'

    def __init__(self,
                 in_channels: int,
                 num_heads: int,
                 kerner_size: int = 1,
                 use_in_conv: bool = False,
                 use_out_conv: bool = False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):

        super().__init__()
        self.num_head = num_heads
        self.in_channels = in_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.kerner_size = kerner_size
        self.use_in_conv = use_in_conv
        self.use_out_conv = use_out_conv
        
        if self.num_head == 1:
            self.ChannelTransformer = CAM()
        else:
            self.ChannelTransformer = CTR(self.num_head)
        
        if self.use_in_conv:
            self.in_conv = ConvModule(
                self.in_channels,
                self.in_channels,
                self.kerner_size,
                padding=1 if self.kerner_size == 3 else 0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        if self.use_out_conv:
            self.out_conv = ConvModule(
                self.in_channels,
                self.in_channels,
                self.kerner_size,
                padding=1 if self.kerner_size == 3 else 0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)     
    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        out = x_input
        
        if self.use_in_conv:
            out = self.in_conv(out)
            
        out = self.ChannelTransformer(out)
        
        if self.use_out_conv:
            out = self.out_conv(out)
            
        return out

@MODELS.register_module()
class SpatialTR(nn.Module):
    _abbr_ = 'SpatialTR_block'

    def __init__(self,
                 in_channels: int,
                 num_heads: int = 8,
                 position_embedding_dim: int = -1,
                 position_magnitude: int = 1,
                 kv_stride: int = 2,
                 q_stride: int = 2):

        super().__init__()

        # hard range means local range for non-local operation
        self.position_embedding_dim = (position_embedding_dim if position_embedding_dim > 0 else in_channels)
        self.position_magnitude = position_magnitude
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.kv_stride = kv_stride
        self.q_stride = q_stride
        self.qk_embed_dim = in_channels // num_heads
        out_c = self.qk_embed_dim * num_heads

        # Q
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_c, kernel_size=1, bias=False)
        self.query_conv.kaiming_init = True

        # K
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_c, kernel_size=1, bias=False)
        self.key_conv.kaiming_init = True

        # V
        self.v_dim = in_channels // num_heads
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=self.v_dim * num_heads, kernel_size=1, bias=False)
        self.value_conv.kaiming_init = True

        # position
        self.appr_geom_fc_x = nn.Linear(self.position_embedding_dim // 2, out_c, bias=False)
        self.appr_geom_fc_x.kaiming_init = True

        self.appr_geom_fc_y = nn.Linear(self.position_embedding_dim // 2, out_c, bias=False)
        self.appr_geom_fc_y.kaiming_init = True

        # parameter
        stdv = 1.0 / math.sqrt(self.qk_embed_dim * 2)
        appr_bias_value = -2 * stdv * torch.rand(out_c) + stdv
        self.appr_bias = nn.Parameter(appr_bias_value)

        # 1x1 
        self.proj_conv = nn.Conv2d(in_channels=self.v_dim * num_heads, out_channels=in_channels, kernel_size=1, bias=True)
        self.proj_conv.kaiming_init = True
        self.gamma = nn.Parameter(torch.zeros(1))

        # downsample
        self.q_downsample = nn.AvgPool2d(kernel_size=1, stride=self.q_stride)
        self.kv_downsample = nn.AvgPool2d(kernel_size=1, stride=self.kv_stride)

        self.init_weights()
        
    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        num_heads = self.num_heads

        # downsample ===================================
        x_q = self.q_downsample(x_input)
        n, _, h, w = x_q.shape # (8, 128, 20, 10)

        x_kv = self.kv_downsample(x_input)
        _, _, h_kv, w_kv = x_kv.shape # (8, 128, 20, 10)

        # Q =============================================
        proj_query = self.query_conv(x_q) # (8, 128, 20, 10)
        proj_query = proj_query.view((n, num_heads, self.qk_embed_dim, h * w)) # (8, 8, 16, 200)
        proj_query = proj_query.permute(0, 1, 3, 2) # (8, 8, 200, 16)

        # k ==============================================
        proj_key = self.key_conv(x_kv) # (8, 128, 20, 10)
        proj_key = proj_key.view((n, num_heads, self.qk_embed_dim, h_kv * w_kv)) # (8, 8, 16, 200)

        # position =========================================
        # x:(10, 10, 64) y:(20, 20, 64)
        position_embed_x, position_embed_y = self.get_position_embedding(
            h, w, h_kv, w_kv, self.q_stride, self.kv_stride,
            x_input.device, x_input.dtype, self.position_embedding_dim)
        # (n, num_heads, w, w_kv, dim)
        position_feat_x = self.appr_geom_fc_x(position_embed_x) # (10, 10, 128)
        position_feat_x = position_feat_x.view(1, w, w_kv, num_heads, self.qk_embed_dim) # (1, 10, 10, 8, 16)
        position_feat_x = position_feat_x.permute(0, 3, 1, 2, 4) # (1, 8, 10, 10, 16)
        position_feat_x = position_feat_x.repeat(n, 1, 1, 1, 1) # (8, 8, 10, 10, 16)
        # position_feat_x: (8, 8, 10, 10, 16)
        
        # (n, num_heads, h, h_kv, dim)
        position_feat_y = self.appr_geom_fc_y(position_embed_y) # (20, 20, 128)
        position_feat_y = position_feat_y.view(1, h, h_kv, num_heads, self.qk_embed_dim) # (1, 20, 20, 8, 16)
        position_feat_y = position_feat_y.permute(0, 3, 1, 2, 4) # (1, 8, 20, 20, 16)
        position_feat_y = position_feat_y.repeat(n, 1, 1, 1, 1) # (8, 8, 20, 20, 16)
        # position_feat_y: (8, 8, 20, 20, 16)

        position_feat_x /= math.sqrt(2) # (8, 8, 10, 10, 16)
        position_feat_y /= math.sqrt(2) # (8, 8, 20, 20, 16)
 
        # init empty energy ========================================
        # (8, 8, 20, 10, 20, 10)
        energy = torch.zeros(n, num_heads, h, w, h_kv, w_kv, dtype=x_input.dtype, device=x_input.device) 

        # parameter ==========================================
        # (8, 8, 1, 16)
        appr_bias = self.appr_bias.view(1, num_heads, 1, self.qk_embed_dim).repeat(n, 1, 1, 1)

        # key content only ====================================
        energy_key = torch.matmul(appr_bias, proj_key) # (8, 8, 1, 200)
        energy_key = energy_key.view(n, num_heads, 1, 1, h_kv, w_kv) # (8, 8, 1, 1, 20, 10)
        energy += energy_key # (8, 8, 20, 10, 20, 10)

        # Q x position ===============================================    
        proj_query_reshape = proj_query.view(n, num_heads, h, w, self.qk_embed_dim) # (8, 8, 20, 10, 16)
        proj_query_reshape = proj_query_reshape.permute(0, 1, 3, 2, 4) # (8, 8, 10, 20, 16)
        position_feat_x_reshape = position_feat_x.permute(0, 1, 2, 4, 3) # (8, 8, 10, 16, 10)
        position_feat_y_reshape = position_feat_y.permute(0, 1, 2, 4, 3) # (8, 8, 20, 16, 20)

        energy_x = torch.matmul(proj_query_reshape, position_feat_x_reshape) # (8, 8, 10, 20, 10)
        energy_x = energy_x.permute(0, 1, 3, 2, 4).unsqueeze(4) # (8, 8, 20, 10, 1, 10)
        
        proj_query_reshape = proj_query_reshape.permute(0, 1, 3, 2, 4) # (8, 8, 20, 10, 16)
        energy_y = torch.matmul(proj_query_reshape, position_feat_y_reshape) # (8, 8, 20, 10, 20)
        energy_y = energy_y.unsqueeze(5) # (8, 8, 20, 10, 20, 1)

        energy += energy_x + energy_y # (8, 8, 20, 10, 20, 10)
        
        # ==========================================================
        energy = energy.view(n, num_heads, h * w, h_kv * w_kv) # (8, 8, 200, 200)

        # attention map ==========================================================
        attention = F.softmax(energy, 3) # (8, 8, 200, 200)
        
        # V ==========================================================
        proj_value = self.value_conv(x_kv) # (8, 128, 20, 10)
        proj_value_reshape = proj_value.view((n, num_heads, self.v_dim, h_kv * w_kv)) # (8, 8, 16, 200)
        proj_value_reshape = proj_value_reshape.permute(0, 1, 3, 2) # (8, 8, 200, 16)

        # attention x V ==========================================================
        out = torch.matmul(attention, proj_value_reshape) # (8, 8, 200, 16)
        out = out.permute(0, 1, 3, 2) # (8, 8, 16, 200)
        out = out.contiguous().view(n, self.v_dim * self.num_heads, h, w) # (8, 128, 20, 10)
    
        # 1x1 ==========================================================    
        out = self.proj_conv(out) # (8, 128, 20, 10)

        # output is downsampled, upsample back to input size
        out = F.interpolate(out, size=x_input.shape[2:], mode='bilinear', align_corners=False) # (8, 128, 40, 20)
            
        out = self.gamma * out + x_input # (8, 128, 40, 20)
            
        return out

    def get_position_embedding(self,
                               h,
                               w,
                               h_kv,
                               w_kv,
                               q_stride,
                               kv_stride,
                               device,
                               dtype,
                               feat_dim,
                               wave_length=1000):
        # the default type of Tensor is float32, leading to type mismatch
        # in fp16 mode. Cast it to support fp16 mode.
        h_idxs = torch.linspace(0, h - 1, h).to(device=device, dtype=dtype)
        h_idxs = h_idxs.view((h, 1)) * q_stride

        w_idxs = torch.linspace(0, w - 1, w).to(device=device, dtype=dtype)
        w_idxs = w_idxs.view((w, 1)) * q_stride

        h_kv_idxs = torch.linspace(0, h_kv - 1, h_kv).to(
            device=device, dtype=dtype)
        h_kv_idxs = h_kv_idxs.view((h_kv, 1)) * kv_stride

        w_kv_idxs = torch.linspace(0, w_kv - 1, w_kv).to(
            device=device, dtype=dtype)
        w_kv_idxs = w_kv_idxs.view((w_kv, 1)) * kv_stride

        # (h, h_kv, 1)
        h_diff = h_idxs.unsqueeze(1) - h_kv_idxs.unsqueeze(0)
        h_diff *= self.position_magnitude

        # (w, w_kv, 1)
        w_diff = w_idxs.unsqueeze(1) - w_kv_idxs.unsqueeze(0)
        w_diff *= self.position_magnitude

        feat_range = torch.arange(0, feat_dim / 4).to(
            device=device, dtype=dtype)

        dim_mat = torch.Tensor([wave_length]).to(device=device, dtype=dtype)
        dim_mat = dim_mat**((4. / feat_dim) * feat_range)
        dim_mat = dim_mat.view((1, 1, -1))

        embedding_x = torch.cat(
            ((w_diff / dim_mat).sin(), (w_diff / dim_mat).cos()), dim=2)

        embedding_y = torch.cat(
            ((h_diff / dim_mat).sin(), (h_diff / dim_mat).cos()), dim=2)

        return embedding_x, embedding_y

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'kaiming_init') and m.kaiming_init:
                kaiming_init(m, mode='fan_in', nonlinearity='leaky_relu',
                    bias=0, distribution='uniform', a=1)

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
    imgs = torch.randn(8, 128, 40, 20)
    gen_attention_block = SpatialTR(in_channels=128, num_heads=8)
    assert gen_attention_block.key_conv.in_channels == 128
    assert hasattr(gen_attention_block, 'appr_bias')
    out = gen_attention_block(imgs)
    print(out.shape)
    assert out.shape == imgs.shape
    
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    flops = FlopCountAnalysis(gen_attention_block, imgs)
    print(flop_count_table(flops))

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