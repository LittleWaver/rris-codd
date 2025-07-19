#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: loss_utils
Author: Waver
"""
import json
import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from src.config import ValidationConfig, AblationConfig


def compute_segmentation_loss(
        logger: logging,
        pred_mask: torch.Tensor,
        target_mask: torch.Tensor
) -> torch.Tensor:

    if target_mask.dtype != torch.long:
        target_mask = target_mask.long()

    try:
        result = F.cross_entropy(pred_mask, target_mask)
    except Exception as e:
        result = torch.tensor(float('nan'), device=pred_mask.device, requires_grad=True)

    return result


def compute_existence_loss(
        logger: logging,
        exist_pred: torch.Tensor,
        exist_label: torch.Tensor
) -> torch.Tensor:

    if exist_pred.dim() > 1 and exist_pred.shape[1] == 1:
        exist_pred_squeezed = exist_pred.squeeze(1)
    else:
        exist_pred_squeezed = exist_pred
    if exist_label.dim() > 1 and exist_label.shape[1] == 1:
        exist_label_squeezed = exist_label.squeeze(1)
    else:
        exist_label_squeezed = exist_label

    try:
        result = F.binary_cross_entropy(exist_pred_squeezed, exist_label_squeezed.float())
    except Exception as e:
        result = torch.tensor(float('nan'), device=exist_pred.device, requires_grad=True)

    return result


def compute_total_loss(
        logger: logging,
        main_mask: torch.Tensor,
        aux_masks: List[torch.Tensor],
        exist_pred: Optional[torch.Tensor],
        target_mask: torch.Tensor,
        exist_label: Optional[torch.Tensor]
) -> torch.Tensor:

    total_loss = torch.tensor(0.0, device=main_mask.device, requires_grad=False)

    use_aux = ValidationConfig.USE_AUXILIARY_MASK
    use_exist = AblationConfig.USE_BCH

    main_loss = compute_segmentation_loss(logger, main_mask, target_mask)
    total_loss += main_loss
    if not torch.isnan(main_loss):
        total_loss = total_loss + main_loss

    if use_aux and aux_masks and len(aux_masks) > 0:
        for i, aux_mask in enumerate(aux_masks):
            aux_loss = compute_segmentation_loss(logger, aux_mask, target_mask)
            total_loss += ValidationConfig.AUXILIARY_MASK_WEIGHT * aux_loss
            if dist.is_initialized():
                torch.distributed.barrier()

    if use_exist and exist_pred is not None and exist_label is not None:
        current_exist_label = exist_label
        if current_exist_label.dim() > 1 and current_exist_label.shape[1] == 1:
            current_exist_label_for_indexing = current_exist_label.squeeze(1).long()
        else:
            current_exist_label_for_indexing = current_exist_label.long()

        has_target = (current_exist_label_for_indexing == 1)
        no_target = (current_exist_label_for_indexing == 0)

        if torch.any(has_target):
            if main_mask[has_target].numel() > 0 and target_mask[has_target].numel() > 0 :
                pos_loss_val = compute_segmentation_loss(
                    logger,
                    main_mask[has_target],
                    target_mask[has_target]
                )
                if not torch.isnan(pos_loss_val):
                    total_loss = total_loss + pos_loss_val

        if main_mask[no_target].numel() > 0 and target_mask[no_target].numel() > 0:
            neg_loss_val = compute_segmentation_loss(
                logger,
                main_mask[no_target],
                target_mask[no_target]
            )
            if not torch.isnan(neg_loss_val):
                total_loss = total_loss + ValidationConfig.NEGATIVE_WEIGHT * neg_loss_val

        exist_loss_val = compute_existence_loss(logger, exist_pred, current_exist_label)
        if not torch.isnan(exist_loss_val):
            total_loss = total_loss + ValidationConfig.EXISTED_PREDICTION_WEIGHT * exist_loss_val

    return total_loss


def save_checkpoint(logger, args, epoch, model, optimizer, lr_scheduler, best=False):

    save_dir = args.exp_dir / "model"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_state={
        'epoch':epoch,
        'model':model.state_dict(),
        'optimizer':optimizer.state_dict(),
        'lr_scheduler':lr_scheduler.state_dict(),
        'args': vars(args)
    }
    base_filename = f"epoch_{epoch}"

    current_model_path = save_dir / f"{base_filename}.pth"
    torch.save(save_state, str(current_model_path))

    if best:
        best_model_path = save_dir / "best.pth"
        torch.save(save_state, str(best_model_path))


def save_best_metrics(logger, args, dict) -> None:

    best_metrics_path = args.exp_dir / 'best_metrics.json'
    try:
        with open(best_metrics_path, 'w', encoding='utf-8') as f:
            json.dump(dict, f, indent=4)
    except Exception as e:
       pass


class AverageMeter:
    def __init__(self, logger) -> None:

        self.logger = logger
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        log_val = val
        if isinstance(val, torch.Tensor):
            val_item = val.item()
            log_val = val_item

        self.val = val
        self.sum += val * n
        self.count += n

        if self.count > 0:
            self.avg = self.sum / self.count
        else:
            self.avg = float('nan')

        self.avg = self.sum / self.count
