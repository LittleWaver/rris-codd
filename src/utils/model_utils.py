#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: model_utils
Author: Waver
"""
import os
import torch
from torch.hub import download_url_to_file
from src.config import SwinConfig, RRISConfig, PathConfig, AblationConfig
from src.model.multi_scale_mask_decoder import MultiScaleMaskDecoder
from src.model.swin_vision_language_encoder import SwinVisionLanguageEncoder


def build_encoder(logger, args, position_encoding):

    swin_vl_encoder = SwinVisionLanguageEncoder(
        logger=logger,img_size=SwinConfig.IMG_SIZE,
        patch_size=SwinConfig.PATCH_SIZE, in_chans=SwinConfig.IN_CHANNELS,
        num_classes=SwinConfig.NUM_CLASSES, embed_dim=SwinConfig.EMBED_DIM,
        depths=SwinConfig.DEPTHS, num_heads=SwinConfig.NUM_HEADS,
        window_size=SwinConfig.WINDOW_SIZE, mlp_ratio=SwinConfig.MLP_RATIO,
        qkv_bias=SwinConfig.QKV_BIAS, qk_scale=SwinConfig.QK_SCALE,
        drop_rate=SwinConfig.DROP_RATE, drop_path_rate=SwinConfig.DROP_PATH_RATE,
        ape=SwinConfig.APE, patch_norm=SwinConfig.PATCH_NORM,
        use_checkpoint=SwinConfig.USE_CHECKPOINT,
        num_mem=AblationConfig.NUM_MEM, num_neg_mem=AblationConfig.NUM_NEG_MEM,
        hidden_dim=RRISConfig.VLCA_HIDDEN_DIM,
        position_encoding=position_encoding)

    load_pretrained_swin(logger, swin_vl_encoder)
    return swin_vl_encoder


def build_decoder(logger, args, position_encoding):

    mask_decoder = MultiScaleMaskDecoder(logger, args, position_encoding)

    return mask_decoder


def print_model_params(logger, model):

    sum = 0
    for name, param in model.named_parameters():
        mul = 1
        for size in param.shape:
            mul *= size
        sum += mul


def load_pretrained_swin(logger, model):

    if not os.path.exists(PathConfig.SWIN_PATH):
        download_url_to_file("https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth",
                             PathConfig.SWIN_PATH)
    checkpoint = torch.load(PathConfig.SWIN_PATH, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            pass
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C2:
            pass
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # delete classification head since we do not use it.
    del state_dict['head.weight']
    del state_dict['head.bias']

    msg = model.load_state_dict(state_dict, strict=False)


def load_checkpoint(args, model_without_ddp, optimizer, lr_scheduler, logger, best=False):
    root_path=args.output
    exp_path=args.exp

    ckpt_path = os.path.join(root_path, exp_path, f'ckpt.pth')
    checkpoint=torch.load(ckpt_path, map_location='cpu')
    msg=model_without_ddp.load_state_dict(checkpoint['model'],strict=False)
    # resume not evaluation
    if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch=checkpoint['epoch']+1
    del checkpoint
    torch.cuda.empty_cache()
