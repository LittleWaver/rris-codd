#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: main
Author: Waver
"""
import logging
import sys
import datetime
import numpy
import torch
import json
from torch.optim import AdamW
from torch.utils.data import DataLoader
from src.dataset.rris_dataset_builder import dataset4train
from src.dataset.rris_dataset_parser import RRISParser
from src.model.rris_model import RRISModel
from src.dataset.rris_dataset import RRISDataset
from src.utils.data_transforms import build_image_transforms
from src.pipeline.validator import Validator
from src.pipeline.trainer import Trainer
from src.utils.args import ArgsManager
from src.utils.logger import LoggerFactory
from src.utils.distributed import init_distributed, create_dist_sampler, \
    create_ddp_model, cleanup_distributed
from src.config import TrainConfig, PathConfig


if __name__ == '__main__':
    """
    python -m src.main --mode train
    python -m src.main --mode eval_illumination
    python -m src.main --mode eval_class
    """

    args = ArgsManager().parse_args()

    exp_dir = PathConfig.OUTPUT_ROOT / \
              f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "log").mkdir(exist_ok=True)
    (exp_dir / "model").mkdir(exist_ok=True)
    args.exp_dir = exp_dir

    logger_factory = LoggerFactory(log_dir=exp_dir / "log")
    logger = logger_factory.create_logger(
        name='main',
        console_level=logging.INFO
    )

    dist_cfg = init_distributed(logger)

    seed = args.seed + dist_cfg['local_rank']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    # cuDNN
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    train_dataset, validation_dataset = dataset4train(args=args, logger=logger)
    train_sampler = create_dist_sampler(train_dataset, dist_cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=TrainConfig.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=0 if sys.platform.startswith('win') else TrainConfig.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
        shuffle=train_sampler is None
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        sampler=train_sampler,
        num_workers=0 if sys.platform.startswith('win') else TrainConfig.NUM_WORKERS,
        shuffle=False
    )

    model = RRISModel(logger, args)
    model = create_ddp_model(model, dist_cfg)

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    backbone_no_decay = []
    backbone_decay = []
    for name, param in model_without_ddp.swin_vl_encoder.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(param)
        else:
            backbone_decay.append(param)
    segmentation_params = [p for p in model_without_ddp.mask_decoder.parameters() if p.requires_grad]

    text_encoder_layers = [
        p
        for i in range(10)
        for p in model_without_ddp.language_encoder.encoder.layer[i].parameters()
        if p.requires_grad
    ]
    params_to_optimize = [
        {'params': backbone_no_decay, 'weight_decay': 0.0},
        {'params': backbone_decay},
        {'params': segmentation_params},
        {'params': text_encoder_layers}
    ]
    optimizer = AdamW(
        params_to_optimize,
        lr=TrainConfig.LR,
        weight_decay=TrainConfig.WEIGHT_DECAY
    )

    num_steps_per_epoch = len(train_loader)
    total_training_steps = num_steps_per_epoch * TrainConfig.EPOCHS
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (1 - step / total_training_steps) ** 0.9
    )

    if args.mode == 'train':
        trainer = Trainer(logger, args, dist_cfg, model, optimizer, lr_scheduler, train_loader, validation_loader)
        trainer.run()

        cleanup_distributed(dist_cfg)

    elif args.mode == 'validation':
        cleanup_distributed(dist_cfg)

    elif args.mode == 'eval_illumination':
        WEIGHT_NAME = 'best'

        WEIGHT_PATH = PathConfig.OUTPUT_ROOT / f'{WEIGHT_NAME}.pth'

        EVAL_SUBSETS = ['testing_dark', 'testing_normal', 'validation_dark', 'validation_normal']

        OUTPUT_DIR = exp_dir / 'illumination'

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if WEIGHT_PATH.exists():

            checkpoint = torch.load(WEIGHT_PATH, map_location='cpu')

            model_without_ddp = model.module if hasattr(model, 'module') else model

            state_dict = checkpoint['model']

            new_state_dict = {}

            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k

                new_state_dict[name] = v

            model_without_ddp.load_state_dict(new_state_dict)

            transformers = build_image_transforms(logger=logger)

            all_results = {}

            for subset in EVAL_SUBSETS:

                eval_dataset = RRISDataset(

                    logger=logger, args=args, split=subset, image_transforms=transformers)

                eval_sampler = create_dist_sampler(eval_dataset, dist_cfg)

                eval_loader = DataLoader(

                    eval_dataset, batch_size=1, sampler=eval_sampler,

                    num_workers=0 if sys.platform.startswith('win') else TrainConfig.NUM_WORKERS,

                    shuffle=False, pin_memory=True)

                validator = Validator(logger, args, dist_cfg, model, eval_loader)

                metrics = validator.run(epoch=10)

                all_results[subset] = metrics

                if dist_cfg['world_size'] > 1:
                    torch.distributed.barrier()

            if dist_cfg['is_main']:

                testing_results = {k: v for k, v in all_results.items() if 'testing' in k}

                validation_results = {k: v for k, v in all_results.items() if 'validation' in k}

                test_output_path = OUTPUT_DIR / f"{WEIGHT_NAME}_testing.json"

                with open(test_output_path, 'w', encoding='utf-8') as f:
                    json.dump(testing_results, f, indent=4)

                val_output_path = OUTPUT_DIR / f"{WEIGHT_NAME}_validation.json"

                with open(val_output_path, 'w', encoding='utf-8') as f:
                    json.dump(validation_results, f, indent=4)

        cleanup_distributed(dist_cfg)

    elif args.mode == 'eval_class':

        WEIGHT_NAME = 'best'
        WEIGHT_PATH = PathConfig.OUTPUT_ROOT / f'{WEIGHT_NAME}.pth'
        EVAL_SPLITS = ['testing', 'validation']
        OUTPUT_DIR = exp_dir / 'class'
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if  WEIGHT_PATH.exists():
            checkpoint = torch.load(WEIGHT_PATH, map_location='cpu')

            model_without_ddp = model.module if hasattr(model, 'module') else model

            state_dict = checkpoint.get('model', checkpoint)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v

            model_without_ddp.load_state_dict(new_state_dict)

            transformers = build_image_transforms(logger=logger)

            temp_parser = RRISParser(logger=logger, args=args, split='testing')
            categories = temp_parser.data['categories']

            for split in EVAL_SPLITS:
                results_per_class = {}
                for category in categories:
                    cat_id = category['id']
                    cat_name = category['name']
                    eval_dataset = RRISDataset(
                        logger=logger, args=args, split=split,
                        image_transforms=transformers,
                        category_id_filter=cat_id
                    )

                    if len(eval_dataset) == 0:
                        results_per_class[cat_name] = {"mIoU": 0.0, "P@0.7": 0.0, "info": "no samples found"}
                        continue

                    eval_sampler = create_dist_sampler(eval_dataset, dist_cfg)
                    eval_loader = DataLoader(
                        eval_dataset, batch_size=1, sampler=eval_sampler,
                        num_workers=0 if sys.platform.startswith('win') else TrainConfig.NUM_WORKERS,
                        shuffle=False, pin_memory=True)

                    validator = Validator(logger, args, dist_cfg, model, eval_loader)
                    metrics = validator.run(epoch=10)

                    miou = metrics.get('mIoU', 0.0)
                    all_precisions = metrics.get('precisions', {})

                    category_result = {"mIoU": miou}
                    category_result.update(all_precisions)

                    results_per_class[cat_name] = category_result

                    if dist_cfg['world_size'] > 1:
                        torch.distributed.barrier()

                if dist_cfg['is_main']:
                    output_filename = f"{WEIGHT_NAME}_{split}.json"
                    output_path = OUTPUT_DIR / output_filename
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(results_per_class, f, indent=4)

        cleanup_distributed(dist_cfg)

    else:
        cleanup_distributed(dist_cfg)

        raise RuntimeError("mode error")
