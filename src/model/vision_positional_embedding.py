#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: vision_positional_embedding
Author: Waver
"""
import math
import torch
from torch import nn
from src.config import RRISConfig, SwinConfig


def build_positional_embedding(logger, args):

    n_steps = RRISConfig.VLCA_HIDDEN_DIM // 2
    if SwinConfig.POSITIONAL_EMBEDDING in ('v2', 'sine'):
        position_embedding = SinePositionalEmbedding(logger, n_steps, normalize=True)
    elif SwinConfig.POSITIONAL_EMBEDDING in ('v3', 'learned'):
        position_embedding = LearnedPositionalEmbedding(logger, n_steps)
    else:
        raise ValueError(f"[RiskZone][build_positional_embedding] "
                         f"not supported {SwinConfig.POSITIONAL_EMBEDDING}")

    return position_embedding


class SinePositionalEmbedding(nn.Module):
    def __init__(self, logger, embed_dim=64, temperature=10000, normalize=False, scale=None):
        super().__init__()

        self.logger = logger
        self.num_pos_feats = embed_dim
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("[RiskZone][SinePositionalEmbedding] "
                             "normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):

        bs, C, H, W = x.shape
        not_mask = torch.ones([bs, H, W], device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos.requires_grad = False
        return pos


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, logger, embed_dim=256):
        super().__init__()

        self.logger = logger
        self.row_embed = nn.Embedding(50, embed_dim)
        self.col_embed = nn.Embedding(50, embed_dim)
        self.reset_parameters()

    def reset_parameters(self):

        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, input_tensor):

        x = input_tensor
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos
