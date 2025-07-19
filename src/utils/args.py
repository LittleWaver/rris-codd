#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: args
Author: Waver
"""
import argparse
import datetime
import json
import logging
import sys
from collections import OrderedDict
from src.config import PathConfig, DatasetConfig, TrainConfig, \
    ValidationConfig, AblationConfig, RRISConfig, BERTConfig, SwinConfig


class ArgsManager:

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='RRIS-CODD')
        self._setup_argument_groups()

    def parse_args(self) -> argparse.Namespace:
        args = self.parser.parse_args()
        self._post_process_ablation_args(args)
        self._save_config(args)
        self.print_config(self.parser, args)
        return args

    def print_config(self, parser, args: argparse.Namespace) -> None:
        for group in parser._action_groups:
            if group.title in ['Positional Arguments', 'options']:
                continue

            for action in group._group_actions:
                arg_name = action.dest
                value = getattr(args, arg_name)
                if isinstance(value, list):
                    value = ', '.join(map(str, value))

    def _setup_argument_groups(self) -> None:
        phase_group = self.parser.add_argument_group('Mode')
        phase_group.add_argument('--mode', default='train',
                                 choices=['train', 'validation', 'test', 'eval_illumination', 'eval_class'])
        experiment_group = self.parser.add_argument_group('Experiment')
        experiment_group.add_argument('--exp_dir', default='exp_unknown')
        experiment_group.add_argument('--seed', type=int, default=20)
        experiment_group.add_argument('--debug', action='store_true')
        dataset_group = self.parser.add_argument_group('Dataset')
        dataset_group.add_argument('--dataset', default='codd',
                                   choices=['codd', 'refcoco', 'refcoco+', 'refcocog', 'rrefcoco'])
        train_group = self.parser.add_argument_group('Training')
        eval_group = self.parser.add_argument_group('Validation/Test')
        phase_group.add_argument('--eval_scope', default='all', choices=['all', 'category'])

        ablation_group = self.parser.add_argument_group('Ablation Study')
        ablation_group.add_argument('--ablation_bch',
                                    type=lambda x: (str(x).lower() == 'true'),
                                    default=None,
                                    help='Ablation: Use Binary Classification Head (True/False)')
        ablation_group.add_argument('--ablation_fpn',
                                    type=lambda x: (str(x).lower() == 'true'),
                                    default=None,
                                    help='Ablation: Use FPN (True/False)')
        ablation_group.add_argument('--ablation_vltf',
                                    type=int, default=None, choices=[1, 2, 3],
                                    help='Ablation: Number of VLTF layers (1, 2, or 3)')
        ablation_group.add_argument('--ablation_memory',
                                    type=lambda x: (str(x).lower() == 'true'),
                                    default=None,
                                    help='Ablation: Use Memory Tokens (True/False)')

    def _save_config(self, args: argparse.Namespace) -> None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config_path = PathConfig.OUTPUT_ROOT / f"args_{timestamp}.json"

        command_line = ' '.join(sys.argv)

        ordered_system_configs = OrderedDict([
            ("command_line", command_line),
            ("TrainConfig", self._get_config_dict(TrainConfig)),
            ("AblationConfig", self._get_config_dict(AblationConfig)),
            ("ValidationConfig", self._get_config_dict(ValidationConfig)),
            ("DatasetConfig", self._get_config_dict(DatasetConfig)),
            ("RRISConfig", self._get_config_dict(RRISConfig)),
            ("SwinConfig", self._get_config_dict(SwinConfig)),
            ("BERTConfig", self._get_config_dict(BERTConfig))
        ])

        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(ordered_system_configs, f, indent=4)
        except Exception as e:
            logging.info(f"{str(e)}")

    def _get_config_dict(self, config_class):
        return {
            key: value for key, value in config_class.__dict__.items()
            if not key.startswith('__') and not callable(value) and not isinstance(value, type)
        }

    def _post_process_ablation_args(self, args: argparse.Namespace) -> None:
        if args.ablation_bch is not None:
            AblationConfig.USE_BCH = args.ablation_bch

        if args.ablation_fpn is not None:
            AblationConfig.USE_FPN = args.ablation_fpn

        if args.ablation_vltf is not None:
            AblationConfig.NUM_VLTF = args.ablation_vltf

        if args.ablation_memory is not None:
            AblationConfig.USE_MEMORY_TOKENS = args.ablation_memory