#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: rris_dataset_builder
Author: Waver
"""
from src.dataset.rris_dataset import RRISDataset
from src.utils.data_transforms import build_image_transforms
from torch.utils.data import Subset
from src.config import TrainConfig


def dataset4train(args, logger):

    transformers = build_image_transforms(
        logger=logger
    )

    train_dataset = RRISDataset(
        logger=logger,
        args=args,
        split='train',
        image_transforms=transformers
    )

    if args.debug:
        train_dataset = Subset(train_dataset, indices=range(TrainConfig.DEBUG_DATASET))

    validation_dataset = RRISDataset(
        logger=logger,
        args=args,
        split='validate',
        image_transforms=transformers
    )

    if args.debug:
        validation_dataset = Subset(validation_dataset, indices=range(TrainConfig.DEBUG_DATASET))

    return train_dataset, validation_dataset
