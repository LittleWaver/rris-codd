#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: statistics
Author: Waver
"""
import json
import time
from pathlib import Path
from typing import Dict


class StatsRecorder:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, logger, args):

        if self._initialized:
            return

        self.logger = logger
        self.args = args

        self.start_time = time.time()

        self.base_dir = self.args.exp_dir

        self._initialized = True
        self.current_epoch = 0

        self._make_dirs()

    def _make_dirs(self):

        if self.args.mode == 'train':
            (self.base_dir / "train").mkdir(parents=True, exist_ok=True)
            (self.base_dir / "validation").mkdir(parents=True, exist_ok=True)
        else:
            (self.base_dir / "validation").mkdir(parents=True, exist_ok=True)

    def _safe_write_json(self, data: Dict, path: Path):

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except (IOError, TypeError) as e:
            return False

    def record_train_summary(self, metrics: Dict):

        path = self.base_dir / "train" / "best_results.json"

        data = {
            "metrics": metrics
        }

        return self._safe_write_json(data, path)

    def record_train_epoch(self, epoch: int, metrics: Dict):

        self.current_epoch = epoch
        epoch_path = self.base_dir / "train"  / f"epoch_{epoch:03d}_metrics.json"

        data = {
            "metrics": metrics
        }

        return self._safe_write_json(data, epoch_path)

    def record_train_batch(self, batch_id: int, metrics: Dict):
        epoch_dir = self.base_dir / "train" / f"epoch_{self.current_epoch:03d}"
        try:
            epoch_dir.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            pass

        batch_path = epoch_dir / f"batch_{batch_id:05d}.json"
        data = {
            "metrics": metrics
        }

        return self._safe_write_json(data, batch_path)

    def record_validation_summary(self, epoch, metrics: Dict):

        path = self.base_dir / "validation" /  f"epoch_{epoch:03d}.json"

        data = {
            "epoch": epoch,
            "metrics": metrics
        }

        return self._safe_write_json(data, path)

    def record_validation_batch(self, epoch, batch_id: int, metrics: Dict):
        epoch_dir = self.base_dir / "validation" / f"epoch_{epoch:03d}"
        try:
            epoch_dir.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            pass

        path = epoch_dir / f"batch_{batch_id:05d}.json"

        data = {
            "epoch": epoch,
            "metrics": metrics
        }

        return self._safe_write_json(data, path)
