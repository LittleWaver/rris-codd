#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: multi_scale_mask_decoder
Author: Waver
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from src.model.deformable_pixel_decoder import MultiScaleDeformableDecoder
from src.model.vision_language_cross_attention import ScaledDotProductAttention
from src.config import SwinConfig, RRISConfig, AblationConfig


class MultiScaleMaskDecoder(nn.Module):
    def __init__(self, logger, args, position_encoding):
        super().__init__()

        self.logger = logger
        self.args = args
        model_name = SwinConfig.NAME
        if model_name == 'swin':
            embed_dim = SwinConfig.EMBED_DIM
            vis_channels = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
            num_heads = SwinConfig.NUM_HEADS
        elif model_name == 'segformer':
            vis_channels = [64, 128, 320, 512]
            num_heads = [1, 2, 5, 8]
        elif model_name == 'convnext':
            vis_channels = [128, 256, 512, 1024]
            num_heads = [1, 2, 4, 8]
        else:
            vis_channels = [256, 512, 1024, 2048]
            num_heads = [4, 8, 16, 32]

        self.cross_attns = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.projects = nn.ModuleList()
        self.mask_convs = nn.ModuleList()
        for i in range(len(vis_channels)):
            if i > 0:
                self.cross_attns.append(ScaledDotProductAttention(self.logger, vis_channels[i],
                                                                  h=num_heads[i], dropout=0.0))
                norm = nn.BatchNorm1d(vis_channels[i])
                nn.init.zeros_(norm.weight)
                self.norms.append(norm)
            self.mask_convs.append(nn.Conv2d(vis_channels[i], 2, 1))
            if i == len(vis_channels) - 1:
                continue
            else:
                channel = vis_channels[i + 1] + vis_channels[i]
                self.projects.append(FeatureProjector(self.logger, channel, vis_channels[i]))

        if AblationConfig.USE_FPN:
            self.pixel_decoder = MultiScaleDeformableDecoder(self.logger, in_channels=vis_channels)
            self.mask_conv = nn.Conv2d(256, 2, 1)
            self.exist_pred_channel = 256
            self.num_heads = 8

        self.exist_pred_channel_index = 1
        self.exist_pred_channel = vis_channels[self.exist_pred_channel_index]
        self.num_heads = num_heads[self.exist_pred_channel_index]

        if AblationConfig.USE_BCH:
            self.src_cross_attn = ScaledDotProductAttention(self.logger, self.exist_pred_channel,
                                                            h=self.num_heads, dropout=0.0)
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.exist_pred = nn.Linear(self.exist_pred_channel, 1)

        self.position_encoding = position_encoding

    def add_positional_embedding(self, tensor, pos):

        return tensor if pos is None else tensor + pos

    def process_pixel_features(self, feature_list):

        for i in range(len(feature_list)):
            B, HW, C = feature_list[i].shape
            H = W = int(math.sqrt(HW))
            feature_list[i] = feature_list[i].view(B, H, W, C).permute(0, 3, 1, 2)  # B, C, H, W
        mask_feature, multi_scale_features = self.pixel_decoder(feature_list)
        return mask_feature

    def forward(self, feature_list, memory_list, lan_attmask):

        if AblationConfig.USE_FPN:

            mask_feature = self.process_pixel_features(feature_list)

            mask_list = []
            if mask_feature is not None:
                generated_mask_logits = self.mask_conv(mask_feature)  # B, 2, H, W
                mask_list.insert(0, generated_mask_logits)

            if mask_feature is not None:
                B, C, H, W = mask_feature.shape
                identity = mask_feature.permute(0, 2, 3, 1).view(B, H * W, C)

            out = feature_list[-1]  # B, HW, C
            mask_list = []
            for i in reversed(range(len(feature_list))):
                if 0 < i <= AblationConfig.NUM_VLTF:
                    identity = out
                    out, att = self.cross_attns[i - 1](out, memory_list[i], memory_list[i])
                    out = identity + self.norms[i - 1](out.permute(0, 2, 1)).permute(0, 2, 1)  # B, HW, C

                B, HW, C = out.shape
                H = W = int(math.sqrt(HW))
                out = out.view(B, H, W, C).permute(0, 3, 1, 2)  # B, C, H, W

                mask = self.mask_convs[i](out)  # B, 2, H, W
                mask_list.insert(0, mask)

                if i > 0:
                    out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
                    next_feature = feature_list[i - 1]
                    B, HW, C = next_feature.shape
                    H = W = int(math.sqrt(HW))
                    next_feature = next_feature.permute(0, 2, 1).view(B, C, H, W)
                    out = torch.cat([out, next_feature], dim=1)
                    out = self.projects[i - 1](out)
                    out = out.permute(0, 2, 3, 1).view(B, H * W, C)

        # --- exist_pred ---
        if AblationConfig.USE_BCH:

            identity_for_exist_pred = None
            if AblationConfig.USE_FPN:
                if 'identity' in locals() and identity is not None:
                    identity_for_exist_pred = identity

            if identity_for_exist_pred is not None:
                mem = memory_list[self.exist_pred_channel_index].detach()  # exist_pred_channel_index = 1
                detached_identity = identity_for_exist_pred.detach()
                exist_feature, att = self.src_cross_attn(detached_identity, mem, mem)

                exist_feature = exist_feature.transpose(1, 2)
                pool_feature = self.avgpool(exist_feature)
                pool_feature = pool_feature.flatten(1)
                exist_pred = torch.sigmoid(self.exist_pred(pool_feature))
            else:
                exist_pred = None
                att = None
        else:
            exist_pred = None
            att = None

        return {
            "mask_list": mask_list,
            "exist_pred": exist_pred if exist_pred is not None else torch.zeros(1, device=feature_list[0].device),
            "att": att if att is not None else torch.zeros(1, device=feature_list[0].device)
        }


class FeatureProjector(nn.Module):
    def __init__(self, logger, C_in, C_out):
        super().__init__()

        self.logger = logger
        self.conv1 = nn.Conv2d(C_in, C_out, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(C_out, C_out, 3, 1, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(C_out)
        self.norm2 = nn.BatchNorm2d(C_out)

    def forward(self, x):

        return F.relu(self.norm2(self.conv2(F.relu(self.norm1(self.conv1(x))))))
