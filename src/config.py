#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: config
Author: Waver
"""
from pathlib import Path


# Path
class PathConfig:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]  # project path
    RAW_ROOT = PROJECT_ROOT / 'raw'
    OUTPUT_ROOT = PROJECT_ROOT / 'output'
    PRETRAINED_DIR = RAW_ROOT / 'pretrained'
    BERT_DIR = PRETRAINED_DIR / 'bert-base-uncased'
    SWIN_PATH = PRETRAINED_DIR / 'swin-base' / 'swin_base_patch4_window12_384_22k.pth'
    IMAGE_DIR = RAW_ROOT / 'image'
    REFS_DIR = RAW_ROOT / 'refs'


# Dataset
class DatasetConfig:
    IMAGE_CROP_SIZE = 480
    IMAGE_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    IMAGE_NORMALIZE_STD = [0.229, 0.224, 0.225]


# Train
class TrainConfig:
    LR = 3e-5  # learning rate: 3e-5 > 0.02 > 0.01 > 0.0001 > 0.00005
    WEIGHT_DECAY = 0.01  # weight of decay
    # num_workers * world_size <= number of CPU core
    NUM_WORKERS = 3  # number of sub-process to load raw
    WORLD_SIZE = 6  # number of process to train, number of GPUs
    BATCH_SIZE = 16
    EPOCHS = 15
    DEBUG_DATASET = 100


# Validation
class ValidationConfig:
    USE_AUXILIARY_MASK = True  # if use auxiliary masks
    AUXILIARY_MASK_WEIGHT = 0.4  # weight of auxiliary masks
    EXISTED_PREDICTION_WEIGHT = 1.0  # weight of existed prediction
    NEGATIVE_WEIGHT = 0.2  # weight of negative sentences，1:2
    THRESHOLD = 0.3


# Ablation Study
class AblationConfig:
    # 1. Binary Classification Head: With/Without
    USE_BCH = True
    # 2. FPN: With/Without
    USE_FPN = True
    # 3. number of VLTFs
    NUM_VLTF = 1
    # 4. Memory Tokens: With/Without
    USE_MEMORY_TOKENS = True
    NUM_MEM = 20  # number of memory tokens
    NUM_NEG_MEM = 10  # number of negative memory tokens


# RRIS
class RRISConfig:
    VLCA_HIDDEN_DIM = 256  # hidden dimension of VisionLanguageCrossAttention


# Language Encoder, BERT
class BERTConfig:
    MAX_TOKENS = 20  # number of max tokens in a sentence
    HIDDEN_SIZE = 768


# Vision Encoder: Swin Transformer
class SwinConfig:
    NAME = 'swin'
    DROP_RATE = 0.0
    DROP_PATH_RATE = 0.1
    NUM_CLASSES = 10
    # origin image resolution(HxW): 1200 x 1920
    # resize: 480 -> 800 -> 960 -> 1024 -> 1152 -> 1280 -> 1344
    IMG_SIZE = 480
    IN_CHANNELS = 3
    PATCH_SIZE = 4
    WINDOW_SIZE = 5
    EMBED_DIM = 128
    DEPTHS = [ 2, 2, 18, 2 ]
    NUM_HEADS = [ 4, 8, 16, 32 ]
    MLP_RATIO = 4.0
    QKV_BIAS = True
    QK_SCALE = None
    APE = False
    PATCH_NORM = True
    USE_CHECKPOINT = False
    POSITIONAL_EMBEDDING = "sine"  # positional embedding type: sine/learned.
