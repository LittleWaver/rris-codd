#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: string_utils
Author: Waver, 542590776@qq.com
"""
import numpy as np
import torch
import functools
import inspect
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from termcolor import colored
from datetime import datetime


def _get_platform_tag() ->str:
    if sys.platform.startswith('win'):
        return 'win'
    elif sys.platform.startswith('linux'):
        return 'linux'
    else:
        return 'other'


def _get_calling_script_path() -> str:
    frame = next(
        (f for f in inspect.stack() if 'logger.py' not in f.filename),
        None
    )
    if frame:
        try:
            full_path = Path(frame.filename).resolve()
            project_root = Path(__file__).resolve().parents[2]
            relative_path = full_path.relative_to(project_root)
            return str(relative_path).replace(os.sep, '.')[:-3]  # 移除.py后缀
        except ValueError:
            return os.path.basename(frame.filename)
    return 'unknown'


class SafeFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, 'module'):
            record.module = _get_calling_script_path()
        if not hasattr(record, 'classname'):
            record.classname = 'Global'
        if not hasattr(record, 'funcName'):
            record.funcName = getattr(record, 'funcName', '<unknown>')
        return super().format(record)


class LoggerFactory:
    def __init__(self, log_dir=None):
        self.log_dir = Path(log_dir)
        self.command_logged = False

    @functools.lru_cache(maxsize=None)
    def create_logger(
            self,
            dist_rank: int = 0,
            name: str = '',
            console_level: int = logging.INFO,
            file_level: int = logging.INFO,
            auto_log_command: bool = True
    ) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        self._clear_handlers(logger)

        if dist_rank == 0:
            console_formatter = self._create_console_formatter()
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(console_level)
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        file_formatter = self._create_file_formatter()
        file_handler = self._create_file_handler(name, dist_rank, file_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        if auto_log_command and not self.command_logged:
            self._log_command_auto(logger)
            self.command_logged = True

        return logger

    def _create_file_handler(
            self,
            name: str,
            dist_rank: int,
            level: int
    ) -> RotatingFileHandler:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"{name}{timestamp}_"
            f"rank{dist_rank}_"
            f"{_get_platform_tag()}.log"
        )
        file_path = self.log_dir / filename

        handler = RotatingFileHandler(
            filename=str(file_path),
            mode='a',
            encoding='utf-8',
            maxBytes=300*1024*1024,
            backupCount=200
        )
        handler.setLevel(level)
        return handler

    def _create_console_formatter(self) -> SafeFormatter:
        color_fmt = (
            colored('[%(asctime)s]', 'green') +
            colored('(%(filename)s %(lineno)d)', 'yellow') +
            colored(' [%(funcName)s]', 'cyan') +
            ': %(levelname)s %(message)s'
        )
        return SafeFormatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S')

    def _create_file_formatter(self) -> SafeFormatter:
        fmt = (
            '[%(asctime)s] '
            '(%(filename)s %(lineno)d)[%(funcName)s]: '
            '%(levelname)s %(message)s'
        )
        return SafeFormatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S')

    def _log_command_auto(self, logger: logging.Logger) -> None:
        cmd = " ".join([f'"{a}"' if ' ' in a else a for a in [sys.executable] + sys.argv])
        logger.info(f"[AUTO_CMD] {cmd}")

    def _clear_handlers(self, logger: logging.Logger) -> None:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)