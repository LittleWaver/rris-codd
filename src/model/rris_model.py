#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: rris_model
Author: Waver
"""
import transformers
from torch import nn
import torch.nn.functional as F
from src.config import SwinConfig
from src.utils.model_utils import build_encoder, build_decoder
from src.model.vision_positional_embedding import build_positional_embedding


class RRISModel(nn.Module):

    def __init__(self, logger, train_args):
        super().__init__()

        self.logger = logger

        self.pretrained_model_name = SwinConfig.NAME

        self.position_encoding = build_positional_embedding(self.logger, train_args)

        self.language_encoder = transformers.BertModel\
            .from_pretrained('raw/pretrained/bert-base-uncased')
        self.language_encoder.pooler = None

        self.swin_vl_encoder = build_encoder(self.logger, train_args, self.position_encoding)

        self.mask_decoder = build_decoder(self.logger, train_args, self.position_encoding)

    def forward(self, image, sentence, sentence_mask):
        
        _, _, H, _ = image.size()

        token_level_text_features = self.language_encoder(sentence, attention_mask=sentence_mask)[0]

        hierarchical_vl_features, vl_memory_buffers, multiscale_vl_attentions,\
            memory2language_weights, pixelwise_feature2memory = \
            self.swin_vl_encoder(image, token_level_text_features, sentence_mask)

        decoder_output = self.mask_decoder(hierarchical_vl_features, vl_memory_buffers, sentence_mask)

        unsampled_mask_predictions = []
        for pred in decoder_output["mask_list"]:
            _, _, h, _ = pred.size()
            assert H % h == 0
            unsampled_mask_predictions.append(F.interpolate(pred, scale_factor=int(H // h),
                                                        mode='bilinear', align_corners=True))

        output_dict = {
            "mask_list": unsampled_mask_predictions,
            "exist_pred": decoder_output.get("exist_pred")
        }

        return output_dict
