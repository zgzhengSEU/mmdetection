import numpy as np
import torch
from torch import nn
from torch.nn import init



class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class SimplifiedScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SimplifiedScaledDotProductAttention, self).__init__()

        self.d_model = d_model
        self.d_k = d_model//h
        self.d_v = d_model//h
        self.h = h

        self.fc_o = nn.Linear(h * self.d_v, d_model)
        self.dropout=nn.Dropout(dropout)



        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = queries.view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = keys.view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = values.view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

class PositionAttentionModule(nn.Module):

    def __init__(self,d_model=512,kernel_size=3,H=7,W=7):
        super().__init__()
        self.cnn=nn.Conv2d(d_model,d_model,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.pa=ScaledDotProductAttention(d_model,d_k=d_model,d_v=d_model,h=1)
    
    def forward(self,x):
        bs,c,h,w=x.shape
        y=self.cnn(x)
        y=y.view(bs,c,-1).permute(0,2,1) #bs,h*w,c
        y=self.pa(y,y,y) #bs,h*w,c
        return y


class ChannelAttentionModule(nn.Module):
    
    def __init__(self,d_model=512,kernel_size=3,H=7,W=7):
        super().__init__()
        self.cnn=nn.Conv2d(d_model,d_model,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.pa=SimplifiedScaledDotProductAttention(H*W,h=1)
    
    def forward(self,x):
        bs,c,h,w=x.shape
        y=self.cnn(x)
        y=y.view(bs,c,-1) #bs,c,h*w
        y=self.pa(y,y,y) #bs,c,h*w
        return y




class DAModule(nn.Module):

    def __init__(self,d_model=512,kernel_size=3,H=7,W=7):
        super().__init__()
        # self.position_attention_module=PositionAttentionModule(d_model=d_model, kernel_size=kernel_size, H=H, W=W)
        self.channel_attention_module=ChannelAttentionModule(d_model=d_model, kernel_size=kernel_size, H=H, W=W)
    
    def forward(self,input):
        bs,c,h,w=input.shape
        # p_out=self.position_attention_module(input)
        c_out=self.channel_attention_module(input)
        # p_out=p_out.permute(0,2,1).view(bs,c,h,w)
        c_out=c_out.view(bs,c,h,w)
        out = c_out
        return out

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from torch import nn

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

# @HEADS.register_module()
# class DAHead(BaseDecodeHead):
#     """Dual Attention Network for Scene Segmentation.

#     This head is the implementation of `DANet
#     <https://arxiv.org/abs/1809.02983>`_.

#     Args:
#         pam_channels (int): The channels of Position Attention Module(PAM).
#     """

#     def __init__(self, pam_channels, **kwargs):
#         super(DAHead, self).__init__(**kwargs)
#         self.pam_channels = pam_channels
#         self.pam_in_conv = ConvModule(
#             self.in_channels,
#             self.channels,
#             3,
#             padding=1,
#             conv_cfg=self.conv_cfg,
#             norm_cfg=self.norm_cfg,
#             act_cfg=self.act_cfg)
#         self.pam = PAM(self.channels, pam_channels)
#         self.pam_out_conv = ConvModule(
#             self.channels,
#             self.channels,
#             3,
#             padding=1,
#             conv_cfg=self.conv_cfg,
#             norm_cfg=self.norm_cfg,
#             act_cfg=self.act_cfg)
#         self.pam_conv_seg = nn.Conv2d(
#             self.channels, self.num_classes, kernel_size=1)

#         self.cam_in_conv = ConvModule(
#             self.in_channels,
#             self.channels,
#             3,
#             padding=1,
#             conv_cfg=self.conv_cfg,
#             norm_cfg=self.norm_cfg,
#             act_cfg=self.act_cfg)
#         self.cam = CAM()
#         self.cam_out_conv = ConvModule(
#             self.channels,
#             self.channels,
#             3,
#             padding=1,
#             conv_cfg=self.conv_cfg,
#             norm_cfg=self.norm_cfg,
#             act_cfg=self.act_cfg)
#         self.cam_conv_seg = nn.Conv2d(
#             self.channels, self.num_classes, kernel_size=1)

#     def pam_cls_seg(self, feat):
#         """PAM feature classification."""
#         if self.dropout is not None:
#             feat = self.dropout(feat)
#         output = self.pam_conv_seg(feat)
#         return output

#     def cam_cls_seg(self, feat):
#         """CAM feature classification."""
#         if self.dropout is not None:
#             feat = self.dropout(feat)
#         output = self.cam_conv_seg(feat)
#         return output

#     def forward(self, inputs):
#         """Forward function."""
#         x = self._transform_inputs(inputs)
#         pam_feat = self.pam_in_conv(x)
#         pam_feat = self.pam(pam_feat)
#         pam_feat = self.pam_out_conv(pam_feat)
#         pam_out = self.pam_cls_seg(pam_feat)

#         cam_feat = self.cam_in_conv(x)
#         cam_feat = self.cam(cam_feat)
#         cam_feat = self.cam_out_conv(cam_feat)
#         cam_out = self.cam_cls_seg(cam_feat)

#         feat_sum = pam_feat + cam_feat
#         pam_cam_out = self.cls_seg(feat_sum)

#         return pam_cam_out, pam_out, cam_out

#     def forward_test(self, inputs, img_metas, test_cfg):
#         """Forward function for testing, only ``pam_cam`` is used."""
#         return self.forward(inputs)[0]

#     def losses(self, seg_logit, seg_label):
#         """Compute ``pam_cam``, ``pam``, ``cam`` loss."""
#         pam_cam_seg_logit, pam_seg_logit, cam_seg_logit = seg_logit
#         loss = dict()
#         loss.update(
#             add_prefix(
#                 super(DAHead, self).losses(pam_cam_seg_logit, seg_label),
#                 'pam_cam'))
#         loss.update(
#             add_prefix(
#                 super(DAHead, self).losses(pam_seg_logit, seg_label), 'pam'))
#         loss.update(
#             add_prefix(
#                 super(DAHead, self).losses(cam_seg_logit, seg_label), 'cam'))
#         return loss

if __name__ == '__main__':
    # input=torch.randn(50,49,512)
    # sa = ScaledDotProductAttention(d_model=512, d_k=512, d_v=512, h=8)
    # output=sa(input,input,input)
    # print(output.shape)
    
    # ssa = SimplifiedScaledDotProductAttention(d_model=512, h=8)
    # output=ssa(input,input,input)
    # print(output.shape)
    
    input=torch.randn(8, 128, 40, 20)
    danet=CTR(num_head=1)
    # print(danet(input).shape)
    
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    flops = FlopCountAnalysis(danet, input)
    print(flop_count_table(flops))