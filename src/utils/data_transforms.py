#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: data_transforms
Author: Waver
"""
import math
import numpy as np
from torchvision.transforms import InterpolationMode
import torch
from torchvision.transforms import functional as F
from src.config import DatasetConfig
from PIL import Image, ImageDraw


def build_image_transforms(logger):
    transforms = [
        AdaptiveInstanceCropperAndResizer(
            logger,
            target_size=DatasetConfig.IMAGE_CROP_SIZE,
            context_factor=0.5,
            min_crop_size=32),
        ToFloatTensor(logger),
        Normalize(logger,
                  mean=DatasetConfig.IMAGE_NORMALIZE_MEAN,
                  std=DatasetConfig.IMAGE_NORMALIZE_STD)
    ]

    return Compose(logger, transforms)


class Compose:
    def __init__(self, logger, transforms):
        self.logger = logger
        self.transforms = transforms

    def __call__(self, image, meta):

        for i, t in enumerate(self.transforms):
            transform_name = type(t).__name__

            pre_mask = meta.get('mask')
            if isinstance(pre_mask, Image.Image):
                pre_np = np.array(pre_mask)
                pre_non_zero = np.count_nonzero(pre_np)
                pre_total = pre_np.size
            elif isinstance(pre_mask, torch.Tensor):
                pre_non_zero = torch.sum(pre_mask > 0.001).item()
                pre_total = pre_mask.numel()
            else:
                pre_non_zero = 0
                pre_total = 0

            image, meta = t(image, meta)

            post_mask = meta.get('mask')
            if isinstance(post_mask, Image.Image):
                post_np = np.array(post_mask)
                post_non_zero = np.count_nonzero(post_np)
                post_total = post_np.size
            elif isinstance(post_mask, torch.Tensor):
                post_non_zero = torch.sum(post_mask > 0.001).item()
                post_total = post_mask.numel()
            else:
                post_non_zero = 0
                post_total = 0

        final_mask = meta['mask']
        final_non_zero = torch.sum(final_mask > 0.001).item()
        final_total = final_mask.numel()

        return image, meta['mask']


def calculate_crop_box(bbox, image_size, context_factor=0.5, min_crop_size=32):
    orig_img_w, orig_img_h = image_size
    box_xmin, box_ymin, box_xmax, box_ymax = bbox
    box_w = box_xmax - box_xmin
    box_h = box_ymax - box_ymin

    if box_w <= 0 or box_h <= 0:
        center_x = box_xmin
        center_y = box_ymin
        crop_side = min_crop_size
    else:
        center_x = box_xmin + box_w / 2
        center_y = box_ymin + box_h / 2
        crop_side_w = box_w * (1 + context_factor)
        crop_side_h = box_h * (1 + context_factor)
        crop_side = max(crop_side_w, crop_side_h, min_crop_size)

    crop_x1 = center_x - crop_side / 2
    crop_y1 = center_y - crop_side / 2
    crop_x2 = center_x + crop_side / 2
    crop_y2 = center_y + crop_side / 2

    actual_crop_x1 = max(0, int(math.floor(crop_x1)))
    actual_crop_y1 = max(0, int(math.floor(crop_y1)))
    actual_crop_x2 = min(orig_img_w, int(math.ceil(crop_x2)))
    actual_crop_y2 = min(orig_img_h, int(math.ceil(crop_y2)))

    if actual_crop_x2 <= actual_crop_x1:
        actual_crop_x2 = actual_crop_x1 + 1
        if actual_crop_x2 > orig_img_w:
            actual_crop_x1 = orig_img_w - 1
            actual_crop_x2 = orig_img_w
    if actual_crop_y2 <= actual_crop_y1:
        actual_crop_y2 = actual_crop_y1 + 1
        if actual_crop_y2 > orig_img_h:
            actual_crop_y1 = orig_img_h - 1
            actual_crop_y2 = orig_img_h

    actual_crop_x1 = min(actual_crop_x1, actual_crop_x2 - 1)
    actual_crop_y1 = min(actual_crop_y1, actual_crop_y2 - 1)

    return (actual_crop_x1, actual_crop_y1, actual_crop_x2, actual_crop_y2)


def crop_and_resize_image(image, crop_box, target_size):
    cropped_image = image.crop(crop_box)
    resized_image = F.resize(cropped_image, target_size, interpolation=InterpolationMode.BILINEAR)
    return resized_image


def crop_and_resize_mask(mask, crop_box, target_size):
    cropped_mask = mask.crop(crop_box)
    resized_mask = F.resize(cropped_mask, target_size, interpolation=InterpolationMode.NEAREST)
    return resized_mask


class AdaptiveInstanceCropperAndResizer:
    """
    Crops the image and mask based on the instance's bounding box,
    then resizes them to the target_size.
    """

    def __init__(self, logger, target_size, context_factor=0.5, min_crop_size=32):
        self.logger = logger
        self.target_size = target_size
        self.context_factor = context_factor  # How much context to include around the bbox
        self.min_crop_size = min_crop_size  # Minimum side length of the cropped area

    def __call__(self, image, meta):

        bbox = meta['bbox']  # [xmin, ymin, xmax, ymax]
        orig_size = (image.width, image.height)
        crop_box = calculate_crop_box(bbox, orig_size, self.context_factor, self.min_crop_size)

        resized_image = crop_and_resize_image(image, crop_box, (self.target_size, self.target_size))

        mask = meta['mask']
        resized_mask = crop_and_resize_mask(mask, crop_box, (self.target_size, self.target_size))

        meta['mask'] = resized_mask

        return resized_image, meta


class ToFloatTensor:

    def __init__(self, logger):
        self.logger = logger

    def __call__(self, image, meta):

        new_image = F.to_tensor(image)

        # new_mask = F.to_tensor(meta['mask']).float()

        mask_pil = meta['mask']
        mask_np = np.array(mask_pil, dtype=np.float32)
        mask_tensor = torch.from_numpy(mask_np)
        new_mask = (mask_tensor > 0.5).float()

        return new_image, {**meta, 'mask': new_mask}


class Normalize:

    def __init__(self, logger, mean, std):
        self.logger = logger
        self.mean = mean
        self.std = std

    def __call__(self, image, meta):

        image = F.normalize(image, mean=self.mean, std=self.std)

        return image, meta


class CenterCropper:

    def __init__(self, logger, crop_size):
        self.logger = logger
        self.crop_size = crop_size

    def __call__(self, image, meta):

        orig_w, orig_h = meta['original_size']
        left = (orig_w - self.crop_size) // 2
        top = (orig_h - self.crop_size) // 2

        new_image = F.crop(image, top, left, self.crop_size, self.crop_size)

        new_polygons = [
            [(x - left, y - top) for x, y in poly]
            for poly in meta['polygons']
        ]

        valid_polygons = []
        for poly in new_polygons:
            filtered = [
                (max(0, min(self.crop_size-1, x)),
                 max(0, min(self.crop_size-1, y)))
                for x, y in poly
            ]
            if len(filtered) >= 3:
                valid_polygons.append(filtered)

        new_mask = self._polygons_to_mask(valid_polygons)

        new_meta = {
            'polygons': valid_polygons,
            'original_size': (self.crop_size, self.crop_size),
            'mask': new_mask
        }

        return new_image, new_meta

    def _polygons_to_mask(self, polygons):

        mask = Image.new('L', (self.crop_size, self.crop_size), 0)
        draw = ImageDraw.Draw(mask)
        for poly in polygons:
            draw.polygon(poly, fill=1)
        return mask


class AdaptiveResizer:

    def __init__(self, logger, target_size):
        self.logger = logger
        self.target_size = target_size

    def __call__(self, image, meta):

        orig_w, orig_h = meta['original_size']
        scale = self.target_size / max(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        new_image = F.resize(image, (new_h, new_w), interpolation=InterpolationMode.BILINEAR)

        scaled_polygons = [
            [(x * scale, y * scale) for x, y in poly]
            for poly in meta['polygons']
        ]

        new_mask = self._generate_mask(scaled_polygons, new_w, new_h)

        new_meta = {
            'polygons': scaled_polygons,
            'original_size': (new_w, new_h),
            'mask': new_mask
        }

        return new_image, new_meta

    def _generate_mask(self, polygons, new_w, new_h):

        mask = Image.new('L', (new_w, new_h), 0)
        draw = ImageDraw.Draw(mask)

        for poly in polygons:
            int_points = [(int(x), int(y)) for x, y in poly]
            safe_points = [
                (max(0, min(new_w-1, x)),  max(0, min(new_h-1, y)))
                for x, y in int_points
            ]
            if len(safe_points) >= 3:
                draw.polygon(safe_points, fill=1)

        return mask


class SmartPadding:

    def __init__(self, logger, target_size):
        self.logger = logger
        self.target_size = target_size

    def __call__(self, image, meta):

        origin_w, origin_h = meta['original_size']
        target = self.target_size

        pad_top = (target - origin_h) // 2
        pad_bottom = (target - origin_h + 1) // 2
        pad_left = (target - origin_w) // 2
        pad_right = (target  - origin_w + 1) // 2
        padding = [pad_left, pad_top, pad_right, pad_bottom]

        new_image = F.pad(image, padding)

        new_polygons = [
            [(x + pad_left, y + pad_top) for x, y in poly]
            for poly in meta['polygons']
        ]

        new_mask = Image.new('L', (self.target_size, self.target_size), 0)
        draw = ImageDraw.Draw(new_mask)
        for poly in new_polygons:
            int_points = [(int(x), int(y)) for x, y in poly]
            safe_points = [
                (max(0, min(self.target_size-1, x)), max(0, min(self.target_size-1, y)))
                for x, y in int_points
            ]
            if len(safe_points) >= 3:
                draw.polygon(safe_points, fill=1)

        new_meta = {
            'polygons': new_polygons,
            'original_size': (target, target),
            'mask': new_mask
        }

        return new_image, new_meta
