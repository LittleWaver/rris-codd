#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: deformable_pixel_decoder
Author: Waver
"""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from mmengine.registry import MODELS
from mmcv.cnn import ConvModule, Conv2d, build_norm_layer
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence, \
    build_positional_encoding, TransformerLayerSequence
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import BaseModule, ModuleList, xavier_init, caffe2_xavier_init, normal_init


class MultiScaleDeformableDecoder(BaseModule):
    def __init__(self,
                 logger,
                 in_channels=[256, 512, 1024, 2048],
                 strides=[4, 8, 16, 32],
                 feat_channels=256,
                 out_channels=256,
                 num_outs=3,
                 norm_cfg=dict(type='GN', num_groups=32),
                 act_cfg=dict(type='ReLU'),
                 encoder=dict(
                     type='DetrTransformerEncoder',
                     num_layers=6,
                     transformerlayers=dict(
                         type='BaseTransformerLayer',
                         attn_cfgs=dict(
                             type='MultiScaleDeformableAttention',
                             embed_dims=256,
                             num_heads=8,
                             num_levels=3,
                             num_points=4,
                             im2col_step=64,
                             dropout=0.0,
                             batch_first=False,
                             norm_cfg=None,
                             init_cfg=None),
                         feedforward_channels=1024,
                         ffn_dropout=0.0,
                         operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                     init_cfg=None),
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.logger = logger
        self.strides = strides
        self.num_input_levels = len(in_channels)
        self.num_encoder_levels = \
            encoder["transformerlayers"]["attn_cfgs"]["num_levels"]
        assert self.num_encoder_levels >= 1, \
            'num_levels in attn_cfgs must be at least one'
        input_conv_list = []
        # from top to down (low to high resolution)
        for i in range(self.num_input_levels - 1,
                       self.num_input_levels - self.num_encoder_levels - 1,
                       -1):
            input_conv = ConvModule(
                in_channels[i],
                feat_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=None,
                bias=True)
            input_conv_list.append(input_conv)
        self.input_convs = ModuleList(input_conv_list)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.postional_encoding = build_positional_encoding(
            positional_encoding)
        # high resolution to low resolution
        self.level_encoding = nn.Embedding(self.num_encoder_levels,
                                           feat_channels)

        self.lateral_convs = ModuleList()
        self.output_convs = ModuleList()
        self.use_bias = norm_cfg is None

        for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1,
                       -1):
            lateral_conv = ConvModule(
                in_channels[i],
                feat_channels,
                kernel_size=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                act_cfg=None)
            output_conv = ConvModule(
                feat_channels,
                feat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)

        self.mask_feature = Conv2d(
            feat_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.num_outs = num_outs
        self.point_generator = DeformablePointGenerator(self.logger, strides)

    def init_weights(self):
        for i in range(0, self.num_encoder_levels):
            xavier_init(
                self.input_convs[i].conv,
                gain=1,
                bias=0,
                distribution='uniform')

        for i in range(0, self.num_input_levels - self.num_encoder_levels):
            caffe2_xavier_init(self.lateral_convs[i].conv, bias=0)
            caffe2_xavier_init(self.output_convs[i].conv, bias=0)

        caffe2_xavier_init(self.mask_feature, bias=0)

        normal_init(self.level_encoding, mean=0, std=1)
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

        for layer in self.encoder.layers:
            for attn in layer.attentions:
                if isinstance(attn, MultiScaleDeformableAttention):
                    attn.init_weights()

    def forward(self, feats):

        batch_size = feats[0].shape[0]
        encoder_input_list = []
        padding_mask_list = []
        level_positional_encoding_list = []
        spatial_shapes = []
        reference_points_list = []

        for i in range(self.num_encoder_levels):  # num_encoder_levels = 3
            level_idx = self.num_input_levels - i - 1  # num_input_levels=len(in_channels)=4. level_idx=3, 2, 1
            feat = feats[level_idx]

            feat_projected = self.input_convs[i](feat)  # B, feat_channels, H, W

            h, w = feat.shape[-2:]
            padding_mask_resized = feat.new_zeros((batch_size, h, w), dtype=torch.bool)
            pos_embed = self.postional_encoding(padding_mask_resized)  # B, feat_channels, H, W

            level_embed = self.level_encoding.weight[i]  # feat_channels
            level_pos_embed = level_embed.view(1, -1, 1, 1) + pos_embed  # B, feat_channels, H, W

            reference_points = self.point_generator.single_level_grid_priors(feat.shape[-2:], level_idx,
                                                                             device=feat.device)  # HW, 2
            factor = feat.new_tensor([[w, h]]) * self.strides[level_idx]
            reference_points_normalized = reference_points / factor

            feat_projected = feat_projected.flatten(2).permute(2, 0, 1)  # HW, B, C
            level_pos_embed = level_pos_embed.flatten(2).permute(2, 0, 1)  # HW, B, C
            padding_mask_resized = padding_mask_resized.flatten(1)  # B, HW

            encoder_input_list.append(feat_projected)
            padding_mask_list.append(padding_mask_resized)
            level_positional_encoding_list.append(level_pos_embed)
            spatial_shapes.append(feat.shape[-2:])
            reference_points_list.append(reference_points_normalized)

        padding_masks = torch.cat(padding_mask_list, dim=1)
        encoder_inputs = torch.cat(encoder_input_list, dim=0)  # Sum(HW_i), B, C
        level_positional_encodings = torch.cat(level_positional_encoding_list, dim=0)  # Sum(HW_i), B, C
        spatial_shapes_tensor = torch.as_tensor(spatial_shapes, dtype=torch.long,
                                         device=encoder_inputs.device)  # num_encoder_levels, 2
        level_start_index = torch.cat((spatial_shapes_tensor.new_zeros((1, )),
                                       spatial_shapes_tensor.prod(1).cumsum(0)[:-1]))  # num_encoder_levels
        reference_points_cat = torch.cat(reference_points_list, dim=0)  # Sum(HW_i), 2
        reference_points_final = reference_points_cat[None, :, None]\
            .repeat(batch_size, 1, self.num_encoder_levels, 1)  # B, Sum(HW_i), num_encoder_levels, 2
        valid_radios = reference_points_final.new_ones((batch_size, self.num_encoder_levels, 2))

        memory = self.encoder(
            query=encoder_inputs,
            key=None, value=None, query_pos=level_positional_encodings,
            key_pos=None, attn_masks=None, key_padding_mask=None,
            query_key_padding_mask=padding_masks,
            spatial_shapes=spatial_shapes_tensor,
            reference_points=reference_points_final,
            level_start_index=level_start_index,
            valid_radios=valid_radios)

        memory = memory.permute(1, 2, 0)  # B, C, Sum(HW_i)

        num_query_per_level = [e[0] * e[1] for e in spatial_shapes]
        outs = torch.split(memory, num_query_per_level, dim=-1)
        # outs_reshaped: list of (B, C, H_i, W_i)
        outs_reshaped = [
            x.reshape(batch_size, -1, spatial_shapes[i][0], spatial_shapes[i][1]) for i, x in enumerate(outs)
        ]

        current_outs_for_fpn = outs_reshaped

        for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1, -1):  # i = 0

            feat_to_fuse = feats[i]  # feats[0]
            lateral_conv_idx = self.num_input_levels - self.num_encoder_levels - 1 - i  # i=0, idx=0

            cur_feat_lateral = self.lateral_convs[lateral_conv_idx](feat_to_fuse)

            # current_outs_for_fpn[-1] 是来自 encoder 的最低分辨率输出 (但经过 reshape), 即对应 feats[1] 的那个
            low_res_feat_to_upsample = current_outs_for_fpn[-1]

            interpolated_feat = F.interpolate(
                low_res_feat_to_upsample,
                size=cur_feat_lateral.shape[-2:],  # 上采样到 cur_feat_lateral 的尺寸
                mode='bilinear',
                align_corners=False)

            fused_feat = cur_feat_lateral + interpolated_feat

            output_fpn_level = self.output_convs[lateral_conv_idx](fused_feat)

            current_outs_for_fpn.append(output_fpn_level)

        multi_scale_features = current_outs_for_fpn[-self.num_outs:]

        highest_res_fpn_feat = current_outs_for_fpn[-1]

        final_mask_feature = self.mask_feature(highest_res_fpn_feat)  # B, out_channels, H, W

        return final_mask_feature, multi_scale_features


class DeformablePointGenerator:

    def __init__(self, logger, strides, offset=0.5):

        self.logger = logger
        self.strides = [_pair(stride) for stride in strides]
        self.offset = offset

    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied"""

        return len(self.strides)

    @property
    def num_base_priors(self):

        return [1 for _ in range(len(self.strides))]

    def _meshgrid(self, x, y, row_major=True):

        yy, xx = torch.meshgrid(y, x)
        if row_major:
            return xx.reshape(-1), yy.reshape(-1)

        else:
            return yy.reshape(-1), xx.reshape(-1)

    def grid_priors(self,
                    featmap_sizes,
                    dtype=torch.float32,
                    device='cuda',
                    with_stride=False):

        assert self.num_levels == len(featmap_sizes)
        multi_level_priors = []
        for i in range(self.num_levels):
            priors = self.single_level_grid_priors(
                featmap_sizes[i],
                level_idx=i,
                dtype=dtype,
                device=device,
                with_stride=with_stride)
            multi_level_priors.append(priors)
        return multi_level_priors

    def single_level_grid_priors(self,
                                 featmap_size,
                                 level_idx,
                                 dtype=torch.float32,
                                 device='cuda',
                                 with_stride=False):

        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = (torch.arange(0, feat_w, device=device) +
                   self.offset) * stride_w
        shift_x = shift_x.to(dtype)
        shift_y = (torch.arange(0, feat_h, device=device) +
                   self.offset) * stride_h
        shift_y = shift_y.to(dtype)
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)

        if not with_stride:
            shifts = torch.stack([shift_xx, shift_yy], dim=-1)
        else:
            stride_w = shift_xx.new_full((shift_xx.shape[0], ),
                                         stride_w).to(dtype)
            stride_h = shift_xx.new_full((shift_yy.shape[0], ),
                                         stride_h).to(dtype)
            shifts = torch.stack([shift_xx, shift_yy, stride_w, stride_h],
                                 dim=-1)
        all_points = shifts.to(device)
        return all_points

    def valid_flags(self, featmap_sizes, pad_shape, device='cuda'):

        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            point_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / point_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / point_stride[0])), feat_w)
            flags = self.single_level_valid_flags((feat_h, feat_w),
                                                  (valid_feat_h, valid_feat_w),
                                                  device=device)
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(self,
                                 featmap_size,
                                 valid_size,
                                 device='cuda'):

        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid

    def sparse_priors(self,
                      prior_idxs,
                      featmap_size,
                      level_idx,
                      dtype=torch.float32,
                      device='cuda'):

        height, width = featmap_size
        x = (prior_idxs % width + self.offset) * self.strides[level_idx][0]
        y = ((prior_idxs // width) % height +
             self.offset) * self.strides[level_idx][1]
        prioris = torch.stack([x, y], 1).to(dtype)
        prioris = prioris.to(device)
        return prioris


@MODELS.register_module()
class DetrTransformerEncoder(TransformerLayerSequence):

    def __init__(self, *args, post_norm_cfg=dict(type='LN'), **kwargs):
        super(DetrTransformerEncoder, self).__init__(*args, **kwargs)
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(
                post_norm_cfg, self.embed_dims)[1] if self.pre_norm else None
        else:
            assert not self.pre_norm, f'Use prenorm in ' \
                                      f'{self.__class__.__name__},' \
                                      f'Please specify post_norm_cfg'
            self.post_norm = None

    def forward(self, *args, **kwargs):
        x = super(DetrTransformerEncoder, self).forward(*args, **kwargs)
        if self.post_norm is not None:
            x = self.post_norm(x)
        return x


@MODELS.register_module()
class SinePositionalEncoding(BaseModule):

    def __init__(self,
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 eps=1e-6,
                 offset=0.,
                 init_cfg=None):
        super(SinePositionalEncoding, self).__init__(init_cfg)
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask):
        mask = mask.to(torch.int)
        not_mask = 1 - mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        B, H, W = mask.size()
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'temperature={self.temperature}, '
        repr_str += f'normalize={self.normalize}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'eps={self.eps})'
        return repr_str


@MODELS.register_module()
class LearnedPositionalEncoding(BaseModule):

    def __init__(self,
                 num_feats,
                 row_num_embed=50,
                 col_num_embed=50,
                 init_cfg=dict(type='Uniform', layer='Embedding')):
        super(LearnedPositionalEncoding, self).__init__(init_cfg)
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = torch.cat(
            (x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(
                1, w, 1)),
            dim=-1).permute(2, 0,
                            1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return pos

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'row_num_embed={self.row_num_embed}, '
        repr_str += f'col_num_embed={self.col_num_embed})'

        return repr_str
