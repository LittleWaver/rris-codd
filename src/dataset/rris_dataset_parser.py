#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: rris_dataset_parser
Author: Waver
"""
import sys
import json
import pickle
from src.config import PathConfig


class RRISParser:
    def __init__(self, logger, args, split):

        self.args = args
        self.logger = logger
        self.split = split

        if self.split == 'train':
            self.reference_dir = PathConfig.REFS_DIR / 'CODD' / 'training'
            self.IMAGE_DIR = PathConfig.IMAGE_DIR / 'CODD' / 'training'
        elif self.split == 'validate' or self.split == 'validation':
            self.reference_dir = PathConfig.REFS_DIR / 'CODD' / 'validation'
            self.IMAGE_DIR = PathConfig.IMAGE_DIR / 'CODD' / 'validation'
        elif self.split == 'validation_dark':
            self.reference_dir = PathConfig.REFS_DIR / 'CODD' / 'validation_dark'
            self.IMAGE_DIR = PathConfig.IMAGE_DIR / 'CODD' / 'validation_dark'
        elif self.split == 'validation_normal':
            self.reference_dir = PathConfig.REFS_DIR / 'CODD' / 'validation_normal'
            self.IMAGE_DIR = PathConfig.IMAGE_DIR / 'CODD' / 'validation_normal'
        elif self.split == 'test' or self.split == 'testing':
            self.reference_dir = PathConfig.REFS_DIR / 'CODD' / 'testing'
            self.IMAGE_DIR = PathConfig.IMAGE_DIR / 'CODD' / 'testing'
        elif self.split == 'testing_dark':
            self.reference_dir = PathConfig.REFS_DIR / 'CODD' / 'testing_dark'
            self.IMAGE_DIR = PathConfig.IMAGE_DIR / 'CODD' / 'testing_dark'
        elif self.split == 'testing_normal':
            self.reference_dir = PathConfig.REFS_DIR / 'CODD' / 'testing_normal'
            self.IMAGE_DIR = PathConfig.IMAGE_DIR / 'CODD' / 'testing_normal'

        all_images, all_annotations, all_categories, all_references = self.init_dataset()
        self.data = {
            'dataset': args.dataset,
            'images': all_images,
            'annotations': all_annotations,
            'categories': all_categories,
            'references': all_references
        }

        self.images_dict = self.build_images_dict()
        self.catetories_dict = self.build_categories_dict()
        self.annotations_dict = self.build_annotations_dict()
        self.references_dict = self.build_references_dict()

    def init_dataset(self):

        # { images[], annotations[], categories[], ... }
        instances = self.load_annotations_file()
        all_images = instances['images']
        all_annotations = instances['annotations']
        all_categories = instances['categories']
        # [ {ref_id, image_id, ann_id, category_id, sentences[], file_name, split ...}, ... ]
        all_references = self.load_references_file()
        return all_images, all_annotations, all_categories, all_references

    def build_images_dict(self):
        """image_id -> image
        {
              "license": -1,
              "file_name": "COCO_train2014_000000098304.jpg",
              "coco_url": "http://mscoco.org/images/98304",
              "height": 424,
              "width": 640,
              "date_captured": "2013-11-21 23:06:41",
              "flickr": "http://farm6.staticflickr.com/5062/5896644212_a326e96ea9_z.jpg",
              "id": 98304
        }
        """

        images_dict = {}
        for image in self.data['images']:
            images_dict[image['id']] = image

        self._save_dict_to_json(images_dict, 'images')

        return images_dict

    def build_categories_dict(self):
        """category_id -> category
        {
            "id": 1,
            "name": "brick"
        }
        """

        categories_dict = {}
        for category in self.data['categories']:
            categories_dict[category['id']] = category

        self._save_dict_to_json(categories_dict, 'categories')

        return categories_dict

    def build_annotations_dict(self):
        """annotation_id -> annotation
        {
            "segmentation": [[267.52, 229.75, 265.6, 226.68, ...]],
            "area": 197.29899999999986,
            "iscrowd": 0,
            "image_id": 98304,
            "bbox": [263.87, 216.88, 21.13, 15.17],
            "category_id": 18,
            "id": 3007
        }
        """

        annotations_dict = {}
        for annotation in self.data['annotations']:
            annotations_dict[annotation['id']] = annotation

        self._save_dict_to_json(annotations_dict, 'annotations')

        return annotations_dict

    def build_references_dict(self):
        """reference_id -> reference
        "1": {
            "file_name": "1.jpg",
            "ann_id": 1,
            "ref_id": 1,
            "image_id": 1,
            "category_id": 1,
            "sentences": [
                {
                    "raw": "the gray brick",
                    "sent_id": 1,
                    "exist": true
                },
                {
                    "raw": "the gray plastics",
                    "sent_id": 4,
                    "exist": false
                }
            ]
        }
        """

        references_dict = {}
        for reference in self.data['references']:
            references_dict[reference['ref_id']] = reference

        self._save_dict_to_json(references_dict, 'references')

        return references_dict

    def get_references_ids(self, split=''):

        references = self.data['references']

        references_ids = [ref['ref_id'] for ref in references]

        return references_ids

    def load_annotations_file(self):

        instances_file_path = self.reference_dir / 'instances.json'

        instances = json.load(open(instances_file_path, 'r'))
        return instances

    def load_references_file(self):

        references_file_path = self.reference_dir / 'refs.p'

        refs = pickle.load(open(references_file_path, 'rb'))
        return refs

    def _save_dict_to_json(self, data_dict, suffix):

        save_dir = PathConfig.RAW_ROOT / 'dict'
        save_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{self.split}_dict_{suffix}.json"
        file_path = save_dir / filename

        with open(file_path, 'w') as f:
            json.dump(data_dict, f, indent=4)
