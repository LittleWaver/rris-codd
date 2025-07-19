#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: codd_refs_generator
Author: Waver
"""
import datetime
import os
import json
import pickle
import random
from collections import defaultdict
from PIL import Image, ImageDraw
import numpy as np
from src.utils.args import ArgsManager
from src.utils.logger import LoggerFactory
from src.config import PathConfig


TARGET_INSTANCES = ["brick", "concrete", "tile", "wood", "pipe", "plastics",
                    "nylon", "plastic bottle", "foam", "stone", "plaster board"]
COLORS = ["red", "gray", "brown", "white", "black", "blue"]
ABSOLUTE_POSITIONS = \
    ["at the top", "at the bottom", "on the left", "on the right", "in the center"]
RELATIVE_POSITIONS = ["above", "below", "to the left of", "to the right of"]


def create_refs_file():
    os.makedirs(REFS_DIR, exist_ok=True)
    if not os.path.exists(REFS_PATH):
        with open(REFS_PATH, "wb") as f:
            pickle.dump([], f)


def load_instances_data():
    with open(INSTANCES_PATH) as f:
        return json.load(f)


def build_category_dict(generator_logger, data):
    generator_logger.info("")

    category_dict = {}
    for category in data["categories"]:
        cid = category["id"]
        if cid not in category_dict:
            category_dict[cid] = category["name"]
    return category_dict


def build_image2instances(generator_logger, data):
    generator_logger.info("")

    image2instances = defaultdict(lambda: {"file_name": "", "instances": []})

    images_map = {img["id"]: img["file_name"] for img in data["images"]}
    categories_map = {cat["id"]: cat["name"] for cat in data["categories"]}

    for ann in data["annotations"]:
        required_fields = ["id", "image_id", "category_id", "segmentation"]
        for field in required_fields:
            if field not in ann:
                continue

        segmentation = ann["segmentation"]
        if not isinstance(segmentation, list):
            segmentation = []

        try:
            normalized_seg = []
            for item in segmentation:
                if isinstance(item, list):
                    normalized_seg.extend([float(x) for x in item])
                else:
                    normalized_seg.append(float(item))
            segmentation = normalized_seg
        except Exception as e:
            segmentation = []

        instance = {
            "ann_id": ann["id"],
            "category_name": categories_map.get(ann["category_id"], "unknown"),
            "bbox": [float(x) for x in ann.get("bbox", [])],
            "segmentation": segmentation
        }

        image2instances[ann["image_id"]]["file_name"] = images_map.get(ann["image_id"], "")
        image2instances[ann["image_id"]]["instances"].append(instance)

    return image2instances


def color_to_rgb(generator_logger, color_name):
    generator_logger.info("")

    color_map = {
        "red": np.array([255, 0, 0]),
        "gray": np.array([128, 128, 128]),
        "brown": np.array([165, 42, 42]),
        "white": np.array([255, 255, 255]),
        "black": np.array([0, 0, 0]),
        "blue": np.array([0, 0, 255])
    }
    return color_map.get(color_name.lower(), np.array([0, 0, 0]))


def get_dominant_color(generator_logger, image_path, segmentation):
    generator_logger.info("")

    try:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
    except Exception as e:
        return random.choice(COLORS)

    mask = Image.new("P", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    try:
        flat_seg = []
        if isinstance(segmentation, list):
            if segmentation and isinstance(segmentation[0], list):
                flat_seg = [coord for sublist in segmentation for coord in sublist]
            else:
                flat_seg = segmentation.copy()

        if len(flat_seg) % 2 != 0:
            return random.choice(COLORS)

        points = [(round(float(flat_seg[i])), round(float(flat_seg[i + 1])))
                  for i in range(0, len(flat_seg), 2)]

        valid_points = [
            (x, y) for x, y in points
            if 0 <= x < width and 0 <= y < height
        ]

        if len(valid_points) < 3:
            return random.choice(COLORS)

        draw.polygon(valid_points, outline=1, fill=1)

        pixels = np.array(img)
        mask_array = np.array(mask).astype(bool)

        if not np.any(mask_array):
            return random.choice(COLORS)

        masked_pixels = pixels[mask_array]
        dominant_color = np.median(masked_pixels, axis=0).astype(int)

        color_distances = {
            color: np.linalg.norm(dominant_color - color_to_rgb(generator_logger, color))
            for color in COLORS
        }

        return min(color_distances, key=color_distances.get)

    except Exception as e:
        return random.choice(COLORS)


def get_absolute_position(generator_logger, bbox, img_size=(640, 480)):
    generator_logger.info("")

    x_center = (bbox[0] + bbox[2] / 2) / img_size[0]
    y_center = (bbox[1] + bbox[3] / 2) / img_size[1]

    if x_center < 0.33: return "on the left"
    if x_center > 0.66: return "on the right"
    if y_center < 0.33: return "at the top"
    if y_center > 0.66: return "at the bottom"
    return "in the center"


def generate_sentences(generator_logger, ann, image_instances, category_dict, sentence_counter):
    generator_logger.info("")

    sentences = []

    target_instance = category_dict[ann["category_id"]]
    image_path = os.path.join(IMAGE_DIR, image_instances["file_name"])

    color = "unknown"
    for instance in image_instances["instances"]:
        if instance["ann_id"] == ann["id"]:
            color = get_dominant_color(generator_logger, image_path, instance["segmentation"])
            break

    num_instances = len(image_instances["instances"])

    # Positive sentences
    if num_instances == 1:
        pos = get_absolute_position(generator_logger, instance["bbox"])
        sentences.extend([
            {"raw": f"the {color} {target_instance}", "sent_id": sentence_counter[0], "exist": True},
            {"raw": f"the {color} {target_instance} {pos}",
             "sent_id": sentence_counter[0] + 1, "exist": True}
        ])
        sentence_counter[0] += 2
    else:
        other = random.choice([i for i in image_instances["instances"] if i["ann_id"] != ann["id"]])
        rel_pos = random.choice(RELATIVE_POSITIONS)
        sentences.extend([
            {"raw": f"the {color} {target_instance}", "sent_id": sentence_counter[0], "exist": True},
            {"raw": f"the {color} {target_instance} {rel_pos} {other['category_name']}",
             "sent_id": sentence_counter[0] + 1, "exist": True}
        ])
        sentence_counter[0] += 2

    # Negative sentences
    if num_instances == 1:
        sentences.extend([
            {"raw": f"the {random.choice([c for c in COLORS if c != color])} {target_instance}",
             "sent_id": sentence_counter[0], "exist": False},
            {"raw": f"the {color} {random.choice([t for t in TARGET_INSTANCES if t != target_instance])}",
             "sent_id": sentence_counter[0] + 1, "exist": False},
            {"raw": f"the {color} {target_instance} "
                    f"{random.choice([p for p in ABSOLUTE_POSITIONS if p != pos])}",
             "sent_id": sentence_counter[0] + 2, "exist": False}
        ])
        sentence_counter[0] += 3
    else:
        sentences.extend([
            {"raw": f"the {random.choice([c for c in COLORS if c != color])} {target_instance}",
             "sent_id": sentence_counter[0], "exist": False},
            {"raw": f"the {color} {random.choice([t for t in TARGET_INSTANCES if t != target_instance])}",
             "sent_id": sentence_counter[0] + 1, "exist": False},
            {"raw": f"the {color} {target_instance} {random.choice([p for p in RELATIVE_POSITIONS if p != rel_pos])} {other['category_name']}",
             "sent_id": sentence_counter[0] + 2, "exist": False},
            {"raw": f"the {color} {target_instance} {rel_pos} {random.choice([t for t in TARGET_INSTANCES if t != other['category_name']])}",
             "sent_id": sentence_counter[0] + 3, "exist": False}
        ])
        sentence_counter[0] += 4

    return sentences


def save_as_json(generator_logger, refs_data):
    json_path = os.path.join(REFS_DIR, "refs.json")
    try:
        def convert_types(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            raise TypeError

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                refs_data,
                f,
                indent=2,
                ensure_ascii=False,
                default=convert_types
            )
    except Exception as e:
        generator_logger.info(f"{str(e)}")


def generator(generator_logger):
    generator_logger.info(f"")

    create_refs_file()
    data = load_instances_data()
    category_dict = build_category_dict(generator_logger, data)

    image2instances = build_image2instances(generator_logger, data)

    refs = []
    ref_id = 1
    sentence_counter = [1]

    for ann in data["annotations"]:
        reference = {
            "file_name": image2instances[ann["image_id"]]["file_name"],
            "ann_id": ann["id"],
            "ref_id": ref_id,
            "image_id": ann["image_id"],
            "category_id": ann["category_id"],
            "sentences": generate_sentences(generator_logger, ann, image2instances[ann["image_id"]], category_dict, sentence_counter)
        }
        refs.append(reference)
        ref_id += 1

    with open(REFS_PATH, "wb") as f:
        pickle.dump(refs, f)

    save_as_json(generator_logger, refs)


"""
training refs.json：
    python -m src.dataset.codd_refs_generator --mode train
validation refs.json：
    python -m src.dataset.codd_refs_generator --mode validation
testing refs.json：
    python -m src.dataset.codd_refs_generator --mode test
"""
if __name__ == "__main__":

    exp_dir = PathConfig.OUTPUT_ROOT / \
              f"generator_refs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger_factory = LoggerFactory(log_dir=exp_dir)
    logger = logger_factory.create_logger()

    args = ArgsManager().parse_args()

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(CURRENT_DIR, '..', '..', 'raw')
    if args.mode == 'train':
        IMAGE_DIR = os.path.join(DATA_DIR, 'image', 'CODD', 'training')
        REFS_DIR = os.path.join(DATA_DIR, 'refs', 'CODD', 'training')
    elif args.mode == 'validation':
        IMAGE_DIR = os.path.join(DATA_DIR, 'image', 'CODD', 'validation')
        REFS_DIR = os.path.join(DATA_DIR, 'refs', 'CODD', 'validation')
    else:
        IMAGE_DIR = os.path.join(DATA_DIR, 'image', 'CODD', 'testing')
        REFS_DIR = os.path.join(DATA_DIR, 'refs', 'CODD', 'testing')
    INSTANCES_PATH = os.path.join(REFS_DIR, "instances.json")
    REFS_PATH = os.path.join(REFS_DIR, "refs.p")

    generator(logger)
