#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: validator
Author: Waver
"""
import numpy as np
import torch
from src.utils.validate_utils import AverageMeter
from src.utils.statistics import StatsRecorder
from src.config import ValidationConfig, AblationConfig
import time


class Validator:
	def __init__(self, logger, args, dist_cfg, model, data_loader):

		self.logger = logger
		self.args = args
		self.dist_cfg = dist_cfg
		self.model = model
		self.data_loader = data_loader
		self.statistics = StatsRecorder(self.logger, self.args)

	def run(self, epoch):

		validate_start_time = time.time()

		positive_iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
		positive_iou_threshold_pass_counts = np.zeros(len(positive_iou_thresholds), dtype=np.int64)

		total_positive_sentence_counts = 0
		total_positive_intersection = AverageMeter(self.logger)
		total_positive_union = AverageMeter(self.logger)
		total_positive_iou = AverageMeter(self.logger)

		total_negative_tn = 0  # True Negative
		total_negative_fp = 0  # False Positive
		total_fp_iou = AverageMeter(self.logger)  # False Positive Intersection over Union

		with torch.no_grad():

			self.model.eval()

			batch_index = 0
			for _, batch_data in enumerate(self.data_loader):
				image, image_mask, sentence, sentence_mask, sentence_exist = batch_data
				"""
				(B, C, H, W)
				(B, H, W), GT of segmentation mask
				(B, seq_len)
				(B, seq_len)
				(B, 1), GT of the existence
				"""

				batch_start_time = time.time()
				batch_positive_intersection = 0.0
				batch_positive_union = 0.0
				batch_positive_iou = 0.0
				batch_negative_tn = 0
				batch_negative_fp = 0
				batch_fp_iou = 0.0

				image = image.cuda(self.dist_cfg['local_rank'], non_blocking=True)
				image_mask = image_mask.cuda(self.dist_cfg['local_rank'], non_blocking=True).long()
				sentence = sentence.cuda(self.dist_cfg['local_rank'], non_blocking=True)
				sentence_mask = sentence_mask.cuda(self.dist_cfg['local_rank'], non_blocking=True)
				sentence_exist = sentence_exist.cuda(self.dist_cfg['local_rank'], non_blocking=True)

				if sentence_exist.item() == 1:  # Positive

					model_output_dict = self.model(image, sentence, sentence_mask)
					current_exist_prediction = model_output_dict.get("exist_pred")
					current_output_masks = model_output_dict.get("mask_list")

					if AblationConfig.USE_BCH:
						if current_exist_prediction is not None:
							exist_pred_scalar = current_exist_prediction.item()
							if exist_pred_scalar < ValidationConfig.THRESHOLD:
								if current_output_masks and current_output_masks[0] is not None:
									current_output_masks = [torch.zeros_like(current_output_masks[0])]

					if current_output_masks and current_output_masks[0] is not None:
						prediction = current_output_masks[0].argmax(1).squeeze(0)
					else:
						prediction = torch.zeros_like(image_mask)

					intersection = torch.sum(torch.mul(prediction, image_mask)).float()
					union = torch.sum(prediction + image_mask).float() - intersection
					iou = intersection / (union + 1e-8)

					batch_positive_intersection = intersection.item()
					batch_positive_union = union.item()
					batch_positive_iou = iou.item()

					total_positive_sentence_counts += 1
					for i, thresh in enumerate(positive_iou_thresholds):
						if iou.item() >= thresh:
							positive_iou_threshold_pass_counts[i] += 1

				else:  # Negative

					model_output_dict = self.model(image, sentence, sentence_mask)
					current_exist_prediction = model_output_dict.get("exist_pred")
					current_output_masks = model_output_dict.get("mask_list")

					if AblationConfig.USE_BCH:
						if current_exist_prediction is not None:
							exist_pred_scalar = current_exist_prediction.item()
							if exist_pred_scalar < ValidationConfig.THRESHOLD:
								if current_output_masks and current_output_masks[0] is not None:
									current_output_masks = [torch.zeros_like(current_output_masks[0])]

					if current_output_masks and current_output_masks[0] is not None:
						prediction = current_output_masks[0].argmax(1).squeeze(0)
						prediction = torch.zeros_like(image_mask)

					if prediction.sum() > 0:  # FP
						batch_negative_fp = 1
						total_negative_fp += 1

						fp_intersection = torch.sum(torch.mul(prediction, image_mask)).float()
						fp_union = torch.sum(prediction + image_mask).float() - fp_intersection
						batch_fp_iou = (fp_intersection / (fp_union + 1e-8)).item()
						total_fp_iou.update(batch_fp_iou)
					else:  # TN
						batch_negative_tn = 1
						total_negative_tn += 1

				if sentence_exist.item() == 1:
					total_positive_intersection.update(batch_positive_intersection)
					total_positive_union.update(batch_positive_union)
					total_positive_iou.update(batch_positive_iou)

				batch_end_time = time.time()
				batch_time = round(int(batch_end_time - batch_start_time) / 60, 3)
				validation_batch_metrics = {
					"batch index": batch_index,
					"time": f"{batch_time} m",
					"sentence_type": "POSITIVE" if sentence_exist.item() == 1 else "NEGATIVE",
					"intersection": batch_positive_intersection if sentence_exist.item() == 1 else 0,
					"union": batch_positive_union if sentence_exist.item() == 1 else 0,
					"iou": batch_positive_iou if sentence_exist.item() == 1 else 0,
					"fp_iou": batch_fp_iou if sentence_exist.item() == 0 else 0,
					"TN": batch_negative_tn,
					"FP": batch_negative_fp
				}
				self.statistics.record_validation_batch(epoch=epoch, batch_id=batch_index,
				                                        metrics=validation_batch_metrics)

				batch_index = batch_index + 1

		validate_end_time = time.time()
		validate_time = round(int(validate_end_time - validate_start_time) / 60, 3)

		iou_str = ' '.join([f'P@{thresh}:{100 * count / total_positive_sentence_counts:.1f}%'
		                    for thresh, count in zip(positive_iou_thresholds, positive_iou_threshold_pass_counts)])

		precisions_at_thresholds = {}
		if total_positive_sentence_counts > 0:
			for thresh, count in zip(positive_iou_thresholds, positive_iou_threshold_pass_counts):
				precisions_at_thresholds[f'P@{thresh}'] = 100 * count / total_positive_sentence_counts
		else:
			for thresh in positive_iou_thresholds:
				precisions_at_thresholds[f'P@{thresh}'] = 0.0

		final_m_iou = 100 * total_positive_iou.avg
		final_o_iou = 100 * (total_positive_intersection.sum / total_positive_union.sum) if total_positive_union.sum > 0 else 0
		final_nsrr = 100 * (total_negative_tn / (total_negative_tn + total_negative_fp + 1e-8))
		final_fp_iou = 100 * total_fp_iou.avg

		validation_final_metrics = {
			"time": f"{validate_time} m",
			"mIoU": final_m_iou,
			"oIoU": final_o_iou,
			"NSRR": final_nsrr,
			"FP-IoU": final_fp_iou,
			"IoU_thresholds": iou_str,
			"precisions": precisions_at_thresholds
		}

		self.statistics.record_validation_summary(epoch=epoch, metrics=validation_final_metrics)

		return {
			"mIoU": final_m_iou,
			"oIoU": final_o_iou,
			"NSRR": final_nsrr,
			"FP-IoU": final_fp_iou,
            "precisions": precisions_at_thresholds
		}
