#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: metrics_utils
Author: Waver
"""
import numpy as np
import torch


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
