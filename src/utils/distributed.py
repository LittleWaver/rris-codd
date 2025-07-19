#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: distributed
Author: Waver
"""
import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def init_distributed(logger=None):
    config = {
        'enabled': False,
        'rank': 0,
        'world_size': 1,
        'local_rank': 0,
        'is_main': True
    }

    if sys.platform.startswith('win'):
        config['local_rank'] = _select_gpu_index()
        _setup_device(config, logger)
        return config

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        try:
            config.update({
                'enabled': True,
                'rank': int(os.environ['RANK']),
                'world_size': int(os.environ['WORLD_SIZE']),
                'local_rank': int(os.environ['LOCAL_RANK'])
            })
            dist.init_process_group(backend='nccl', init_method='env://')
            config['is_main'] = (config['rank'] == 0)
            _setup_device(config, logger)
            return config
        except Exception as e:
            if logger:
                logger.info(f"{str(e)}")
            raise

    config['local_rank'] = _select_gpu_index()
    _setup_device(config, logger)
    return config


def _select_gpu_index():
    if torch.cuda.device_count() == 0:
        return 0

    mem_info = []
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        free_mem = torch.cuda.mem_get_info()[0]
        mem_info.append((i, free_mem))

    mem_info.sort(key=lambda x: x[1], reverse=True)
    return mem_info[0][0]


def _setup_device(config, logger):

    torch.cuda.set_device(config['local_rank'])
    device = torch.cuda.current_device()

    total_mem = torch.cuda.get_device_properties(device).total_memory
    if logger:
        free_mem = torch.cuda.mem_get_info()[0]


def create_ddp_model(model, config):
    if config['enabled']:
        return DDP(model.cuda(), device_ids=[config['local_rank']], find_unused_parameters=True)
    return model.cuda()


def create_dist_sampler(dataset, config, shuffle=True):
    if config['enabled']:
        return torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=config['world_size'],
            rank=config['rank'],
            shuffle=shuffle
        )
    return None


def cleanup_distributed(config):
    if config['enabled'] and dist.is_initialized():
        dist.destroy_process_group()