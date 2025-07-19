#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: trainer
Author: Waver
"""
import time
import json
import torch
from src.pipeline.validator import Validator
from src.utils.train_utils import compute_total_loss
from src.utils.train_utils import save_checkpoint, save_best_metrics
from src.utils.train_utils import AverageMeter
from src.utils.statistics import StatsRecorder
from src.config import TrainConfig


class Trainer:
    def __init__(self, logger, args, dist_cfg, model, optimizer, lr_scheduler, train_loader, eval_loader):

        self.device = torch.device(f"cuda:{dist_cfg['local_rank']}" if torch.cuda.is_available() else 'cpu')

        self.logger = logger
        self.args = args
        self.dist_cfg = dist_cfg
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.statistics = StatsRecorder(self.logger, self.args)

        self.is_distributed = dist_cfg["world_size"] > 1

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{dist_cfg['local_rank']}")
        else:
            self.device = torch.device('cpu')

    def run(self):

        train_start_time = time.time()
        train_best_metrics = {
            "epoch": -1,
            "FP-IoU": -1.0,
            "NSRR": -1.0,
            "mIoU": -1.0,
            "oIoU": -1.0
        }

        for epoch in range(0, TrainConfig.EPOCHS):

            epoch_start_time = time.time()
            if self.is_distributed:
                self.train_loader.sampler.set_epoch(epoch)

            self.train1epoch(epoch)

            self.args.mode = "validation"
            validator = Validator(self.logger, self.args, self.dist_cfg, self.model, self.eval_loader)
            epoch_metrics = validator.run(epoch)

            epoch_end_time = time.time()
            train_epoch_time = round(int(epoch_end_time - epoch_start_time) / 60, 3)
            train_epoch_metrics = {
                "time": f"{train_epoch_time} minute",
                "best metrics": epoch_metrics
            }

            self.statistics.record_train_epoch(epoch=epoch, metrics=train_epoch_metrics)

            if not self.dist_cfg['is_main']:
                torch.distributed.barrier()
            if self.dist_cfg['is_main']:
                if epoch_metrics["FP-IoU"] > train_best_metrics["FP-IoU"]:
                    train_best_metrics = epoch_metrics
                    save_checkpoint(
                        self.logger,
                        self.args,
                        epoch=epoch,
                        model=self.model.module if hasattr(self.model, "module") else self.model,
                        optimizer=self.optimizer,
                        lr_scheduler=self.lr_scheduler,
                        best=True)
                    best_metrics = {
                        "epoch": epoch,
                        "best metrics": train_best_metrics
                    }
                    save_best_metrics(logger=self.logger, args=self.args, dict=best_metrics)

        train_end_time = time.time()
        train_time = round(int(train_end_time - train_start_time) / 60, 3)
        train_final_metrics = {
            "time": f"{train_time} minute",
            "best metrics": train_best_metrics
        }

        self.statistics.record_train_summary(metrics=train_final_metrics)

    def train1epoch(self, epoch):

        epoch_start_time = time.time()
        epoch_metrics = {
            "epoch": epoch,
            "FP-IoU": -1.0,
            "NSRR": -1.0,
            "mIoU": -1.0,
            "oIoU": -1.0
        }
        epoch_loss = AverageMeter(self.logger)

        batch_losses = []

        for idx, (img, target, emb, att_mask, exist) in enumerate(self.train_loader):
            """
            (B, C, H, W)
            (B, H, W), GT mask for segmentation
            (B, seq_len)
            (B, seq_len)
            (B, 1), GT for existence
            """

            self.train1batch(idx, img, target, emb, att_mask, exist, epoch, epoch_loss, batch_losses)

        if not self.dist_cfg['is_main']:
            torch.distributed.barrier()
        if self.dist_cfg['is_main']:
            save_checkpoint(
                self.logger,
                self.args,
                epoch=epoch,
                model=self.model.module if hasattr(self.model, "module") else self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                best=False)
            epoch_end_time = time.time()
            epoch_time = round(int(epoch_end_time - epoch_start_time) / 60, 3)
            train_epoch_metrics = {
                "time": f"{epoch_time} m",
                "metrics": epoch_metrics,
                "mean loss": epoch_loss.avg
            }

            if batch_losses:
                loss_file = self.args.exp_dir / "train" / f"epoch_{epoch:03d}_loss.json"
                try:
                    with open(loss_file, 'w', encoding='utf-8') as f:
                        json.dump(batch_losses, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    self.logger.error(f"{str(e)}")

        epoch_loss.reset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train1batch(self, idx, img, target, emb, att_mask, exist, epoch, epoch_loss, batch_losses):

        batch_start_time = time.time()
        batch_size = target.size(0)
        batch_loss = -1.0

        target = target.long()
        emb = emb.squeeze(1)
        att_mask = att_mask.squeeze(1)
        img = img.cuda(self.dist_cfg['local_rank'], non_blocking=True)
        target = target.cuda(self.dist_cfg['local_rank'], non_blocking=True)
        emb = emb.cuda(self.dist_cfg['local_rank'], non_blocking=True)
        att_mask = att_mask.cuda(self.dist_cfg['local_rank'], non_blocking=True)
        exist = exist.cuda(self.dist_cfg['local_rank'], non_blocking=True)
        model_output_dict  = self.model(img, emb, att_mask)

        main_mask_from_model = None
        aux_masks_from_model = []
        exist_pred_from_model = None
        loss_calculated = False

        if model_output_dict is not None:
            main_mask_from_model = model_output_dict["mask_list"][0]
            aux_masks_from_model = model_output_dict["mask_list"][1:]

            if "exist_pred" in model_output_dict and model_output_dict["exist_pred"] is not None:
                exist_pred_from_model = model_output_dict["exist_pred"]
            else:
                exist_pred_from_model = torch.zeros((batch_size, 1), device=img.device)

        loss = compute_total_loss(self.logger,
                                  main_mask_from_model,
                                  aux_masks_from_model,
                                  exist_pred_from_model,
                                  target,
                                  exist)
        loss_calculated = True
        if not loss_calculated:
            loss = torch.tensor(float('nan'), device=img.device, requires_grad=True)

        self.optimizer.zero_grad()
        loss.backward()

        # max_norm = 1.0  # 1.0, 5.0, 10.0
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)

        self.optimizer.step()
        self.lr_scheduler.step()
        torch.cuda.synchronize()
        epoch_loss.update(loss.item(), batch_size)

        if not self.dist_cfg['is_main']:
            torch.distributed.barrier()
        if self.dist_cfg['is_main']:
            batch_end_time = time.time()
            batch_time = round(int(batch_end_time - batch_start_time) / 60, 3)
            batch_loss = loss.item()
            train_batch_metrics = {
                "loss": batch_loss,
                "batch_index": f"batch {idx}",
                "batch_size": batch_size,
                "time": f"{batch_time} m"
            }

            batch_losses.append(train_batch_metrics)
