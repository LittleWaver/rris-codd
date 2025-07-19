#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: rris_dataset
Author: Waver
"""
import torch.utils.data as data
import torch 
import numpy as np 
from PIL import Image, ImageDraw
import transformers
from src.config import BERTConfig, PathConfig
from src.dataset.rris_dataset_parser import RRISParser


class RRISDataset(data.Dataset):
    def __init__(self,
                 logger=None,
                 args=None,
                 split='train',
                 image_transforms=None,
                 category_id_filter=None) -> None:

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.args = args
        self.logger = logger
        self.split = split
        self.image_transforms = image_transforms
        self.category_id_filter = category_id_filter

        # Tokenizer
        self.tokenizer = transformers.BertTokenizer.from_pretrained(PathConfig.BERT_DIR)

        # RRISParser
        self.rris_dataset_parser = RRISParser(logger, args, self.split)

        # Dataset Item List
        self.sentence_items = []
        for ref_id in self.rris_dataset_parser.get_references_ids(split=self.split):
            reference = self.rris_dataset_parser.references_dict[ref_id]
            if self.category_id_filter is not None and reference['category_id'] != self.category_id_filter:
                continue
            for sent_idx, _ in enumerate(reference['sentences']):
                self.sentence_items.append({'ref_id': ref_id, 'sent_idx': sent_idx})

        # Tokenized sentences and sentences masks
        self.tokenized_sentences, self.tokenized_sentences_masks = self._precompute_tokenized_data()

    def __len__(self):
        return len(self.sentence_items)

    def __getitem__(self, index):

        item_info = self.sentence_items[index]
        refer_id = item_info['ref_id']
        sent_idx = item_info['sent_idx']

        refer = self.rris_dataset_parser.references_dict[refer_id]
        """reference_id -> reference
        {
            "sent_ids": [0, 1, 2],
            "file_name": "COCO_train2014_000000581857_16.jpg",
            "ann_id": 1719310,
            "ref_id": 0,
            "image_id": 581857,
            "split": "training",
            "sentences": [
                {
                    "tokens": ["navy", "blue", "shirt"],
                    "raw": "navy blue shirt", "sent_id": 0, "sent": "navy blue shirt",
                    "exist": true
                },
                ...
            ],
            "category_id": 1
        }
        """
        img_id = refer['image_id']
        image = self.rris_dataset_parser.images_dict[img_id]
        cate_id = refer['category_id']
        cate = self.rris_dataset_parser.catetories_dict[cate_id]
        ann_id = refer['ann_id']
        ann = self.rris_dataset_parser.annotations_dict[ann_id]

        # sentence; sentence_mask; sentence_exist
        sentence4model = self.tokenized_sentences[index]
        sentence_mask4model = self.tokenized_sentences_masks[index]
        sentence_exist4model = self.validate_sentence_grounding(refer, sent_idx)  # [1.0]/[0.0]

        # image; origin mask
        try:
            image_rgb = Image.open(self.rris_dataset_parser.IMAGE_DIR / image['file_name']).convert("RGB")
        except Exception as e:
            self.logger.info("{str(e)}")
            raise FileNotFoundError
        binary_mask_pil, polygons = self._generate_vector_mask(ann, image)

        # Image Transforms
        original_meta = {
            'bbox': ann['bbox'],
            'original_size': image_rgb.size,
            'polygons': polygons,
            'mask': binary_mask_pil
        }
        image4model, binary_mask_transformed = self.image_transforms(image_rgb, original_meta)

        # ground truth(GT) mask
        image_mask4model = binary_mask_transformed.squeeze(0)
        if self.args.mode == 'train':
                if sentence_exist4model.item() == 0.0:
                    image_mask4model = torch.zeros_like(image_mask4model)

        # (C, H, W); (H, W), GT of segmentation mask; (seq_len); (seq_len); (0)/(1), GT of the existence
        return image4model, image_mask4model, sentence4model, sentence_mask4model, \
               sentence_exist4model

    def _precompute_tokenized_data(self):

        tokenized_sentences_list = []
        tokenized_sentences_masks_list = []

        for item_info in self.sentence_items:
            ref_id = item_info['ref_id']
            sent_idx = item_info['sent_idx']

            ref = self.rris_dataset_parser.references_dict[ref_id]
            sentence_data = ref['sentences'][sent_idx]
            sentence_raw = sentence_data['raw']

            encoded_dict = self.tokenizer.encode_plus(
                sentence_raw,
                add_special_tokens=True,
                max_length=BERTConfig.MAX_TOKENS,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            padded_input_ids = encoded_dict['input_ids'].squeeze(0)
            attention_mask = encoded_dict['attention_mask'].squeeze(0)

            tokenized_sentences_list.append(padded_input_ids)
            tokenized_sentences_masks_list.append(attention_mask)

        return tokenized_sentences_list, tokenized_sentences_masks_list

    def validate_sentence_grounding(self, ref, sent_index):

        exist_flag = True
        if "sentences" in ref and sent_index < len(ref["sentences"]):
            sentence_info = ref["sentences"][sent_index]
            if "exist" in sentence_info:
                exist_flag = sentence_info["exist"]

        output = torch.tensor([float(exist_flag)], dtype=torch.float32)

        return output

    def _generate_vector_mask(self, ann, image_info):

        width, height = image_info['width'], image_info['height']
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)

        polygons = ann['segmentation']
        valid_polygons = []

        for poly in polygons:
            if len(poly) % 2 != 0:
                truncated_length = len(poly) - 1
                poly = poly[:truncated_length]
            abs_points = []
            for i in range(0, len(poly), 2):
                if i + 1 >= len(poly):
                    continue
                x = poly[i]
                y = poly[i + 1]
                abs_points.append((x, y))
            if len(abs_points) >= 3:
                valid_polygons.append(abs_points)
                draw.polygon(abs_points, fill=1)

        return mask, valid_polygons

    def _generate_zero_mask(self, img_size):

        zero_mask = Image.new("L", img_size, 0)

        return zero_mask
