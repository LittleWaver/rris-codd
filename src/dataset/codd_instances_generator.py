#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: codd_instances_generator
Author: Waver
"""
import datetime
import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from src.config import PathConfig
from src.utils.args import ArgsManager
from src.utils.logger import LoggerFactory


def generator(generator_logger, args):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', '..', 'raw')

    if args.mode == 'train':
        codd_dir = os.path.join(data_dir, 'image', 'CODD', 'training')
        refs_dir = os.path.join(data_dir, 'refs', 'CODD', 'training')
    elif args.mode == 'validation':
        codd_dir = os.path.join(data_dir, 'image', 'CODD', 'validation')
        refs_dir = os.path.join(data_dir, 'refs', 'CODD', 'validation')
    else:
        codd_dir = os.path.join(data_dir, 'image', 'CODD', 'testing')
        refs_dir = os.path.join(data_dir, 'refs', 'CODD', 'testing')

    json_path = os.path.join(refs_dir, 'instances.json')

    Path(refs_dir).mkdir(parents=True, exist_ok=True)

    data = {
        "info": {
            "description": "The CODD dataset is a meticulously curated collection of images and "
                           "annotations designed to facilitate the development and benchmarking of "
                           "both bounding box and instance segmentation detection models for "
                           "Construction and Demolition Waste (CDW) sorting. CODD features "
                           "ten distinct CDW categories, including bricks, concrete, tiles, wood, pipes, "
                           "plastics, general waste, foaming insulation, stones, and plaster boards. "
                           "The dataset was carefully acquired from a recycling facility in Cyprus, "
                           "capturing the diverse characteristics of CDW in their natural state. "
                           "A total of 3,129 high-resolution images (1920 × 1200 × 3, RGB) "
                           "containing 16,545 annotated samples make up this comprehensive resource. "
                           "The dataset is divided into training, validation, and testing subsets, "
                           "with the option for users to exercise discretion in the use of the validation "
                           "set within their specific research framework. All annotations are provided in "
                           "the standardized PASCAL VOC XML format, including both bounding box "
                           "coordinates and polygon coordinates for precise object segmentation detection.",
            "url": "https://data.mendeley.com/datasets/wds85kt64j/3",
            "version": "3.0",
            "publishd_date": "2023-10-05",
            "contributor": "Demetris Demetriou, Pavlos Mavromatidis, Michael Petrou, Demetris Nicolaides",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/"
        },
        "images": [],
        "annotations": [],
        "categories": []
    }

    category_dict = {}
    image_id = 0
    annotation_id = 0

    for xml_file in Path(codd_dir).glob('*.xml'):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            filename = root.find('filename').text
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            depth = int(size.find('depth').text)

            image_id += 1
            data['images'].append({
                "id": image_id,
                "file_name": filename,
                "width": width,
                "height": height,
                "depth": depth
            })

            for obj in root.findall('object'):
                name = obj.find('name').text.strip()
                if name not in category_dict:
                    category_id = len(category_dict) + 1
                    category_dict[name] = category_id
                    data['categories'].append({
                        "id": category_id,
                        "name": name
                    })
                current_category_id = category_dict[name]

                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                bbox = [xmin, ymin, xmax, ymax]

                segmentation = []
                polygon = obj.find('polygon')
                if polygon is not None:
                    elements = list(polygon)
                    if len(elements) % 2 != 0:
                        continue

                    polygon_points = []
                    for i in range(0, len(elements), 2):
                        try:
                            x = int(round(float(elements[i].text)))
                            y = int(round(float(elements[i + 1].text)))
                            polygon_points.extend([x, y])
                        except (ValueError, IndexError) as e:
                            continue

                    if len(polygon_points) >= 4:
                        if (polygon_points[0] != polygon_points[-2]) or \
                                (polygon_points[1] != polygon_points[-1]):
                            polygon_points.extend([polygon_points[0], polygon_points[1]])
                        segmentation.append(polygon_points)

                annotation_id += 1
                data['annotations'].append({
                    "id": annotation_id,
                    "bbox": bbox,
                    "area": 0,
                    "segmentation": segmentation,
                    "image_id": image_id,
                    "category_id": current_category_id
                })

        except Exception as e:
            generator_logger.info(f"Error processing {xml_file}: {str(e)}")
            continue

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    normalized_path = os.path.normpath(json_path)
    generator_logger.info(f"Successfully  generated： {normalized_path}")
    generator_logger.info(f"info： {data['info']}")
    generator_logger.info(f"images length：{len(data['images'])}")
    generator_logger.info(f"annotations length：{len(data['annotations'])}")
    generator_logger.info(f"categories length：{len(data['categories'])}")


"""
training instances.json：
    python -m src.dataset.codd_instances_generator --mode train
validation instances.json：
    python -m src.dataset.codd_instances_generator --mode validation
testing instances.json：
    python -m src.dataset.codd_instances_generator --mode test
"""
if __name__ == "__main__":

    exp_dir = PathConfig.OUTPUT_ROOT / \
              f"generator_instances_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger_factory = LoggerFactory(log_dir=exp_dir)
    logger = logger_factory.create_logger()
    
    args = ArgsManager().parse_args()

    generator(logger, args)
