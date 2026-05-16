# Data Preparation Guide

This document provides a step-by-step guide to preparing the BDD100K weather-balanced dataset for training and evaluation with the WeatherAdaptive-Perception project.

---

## Table of Contents

[zh_CN 中文版](data_preparation_cn.md)

- [1. BDD100K Dataset Overview](#1-bdd100k-dataset-overview)
- [2. Download](#2-download)
- [3. Raw Data Structure](#3-raw-data-structure)
- [4. Label Format Explanation and Conversion](#4-label-format-explanation-and-conversion)
- [5. Weather Label Extraction](#5-weather-label-extraction)
- [6. Weather-Balanced Sampling](#6-weather-balanced-sampling)
- [7. Final Directory Structure](#7-final-directory-structure)
- [8. Dataset Configuration File](#8-dataset-configuration-file)
- [9. FAQ](#9-faq)

---

## 1. BDD100K Dataset Overview

[BDD100K](https://www.bdd100k.com/) is a large-scale driving dataset released by UC Berkeley, containing 100,000 driving scene images across diverse weather, time-of-day, and scene conditions. It is well-suited for research on environment-adaptive object detection models.

Key attribute annotations in the dataset:

| Attribute Category | Possible Values |
|-------------------|-----------------|
| Weather | clear, partly cloudy, overcast, rainy, snowy, foggy |
| Time of Day | daytime, night, dawn/dusk |
| Scene | city street, highway, residential, parking lot, gas station, tunnel |

Object detection annotations cover 10 categories: pedestrian, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign.

This project performs weather-balanced sampling from BDD100K to construct a subset of approximately 15k images for training.

---

## 2. Download

### 2.1 Official Source

Visit the [BDD100K official website](https://www.bdd100k.com/), register an account, and download the following files:

- **100K Images** (`bdd100k_images_100k.zip`, ~6.5GB) — All images
- **Detection Labels** (`bdd100k_labels_release.zip`, ~100MB) — Detection annotation JSON files

### 2.2 Kaggle (Alternative)

If the official download is slow, you can search for BDD100K mirrors on Kaggle where some users have uploaded copies.

### 2.3 After Downloading

Extract both zip files into the same root directory:

```bash
mkdir -p datasets/bdd100k
cd datasets/bdd100k

# Extract images
unzip bdd100k_images_100k.zip

# Extract labels
unzip bdd100k_labels_release.zip
```

---

## 3. Raw Data Structure

After extraction, the directory structure is as follows:

```
datasets/bdd100k/
├── images/
│   └── 100k/
│       ├── train/          # 70,000 training images (1280×720, .jpg)
│       ├── val/            # 10,000 validation images
│       └── test/           # 20,000 test images (no labels)
└── labels/
    └── bdd100k_labels_images_train.json    # Training set annotations
    └── bdd100k_labels_images_val.json      # Validation set annotations
```

### 3.1 Label JSON Structure

Each JSON file contains a list where each element corresponds to one image:

```json
{
    "name": "b1c66a42-6f7d68ca.jpg",
    "attributes": {
        "weather": "clear",
        "scene": "city street",
        "timeofday": "daytime"
    },
    "labels": [
        {
            "category": "car",
            "box2d": {
                "x1": 1022.15,
                "y1": 387.15,
                "x2": 1248.78,
                "y2": 546.19
            }
        },
        {
            "category": "traffic sign",
            "box2d": {
                "x1": 471.62,
                "y1": 281.39,
                "x2": 510.58,
                "y2": 326.65
            }
        }
    ]
}
```

The `attributes.weather` field is the key attribute used for weather-balanced sampling in this project.

---

## 4. Label Format Explanation and Conversion

### 4.1 BDD100K Original Format

BDD100K uses absolute pixel coordinates in `(x1, y1, x2, y2)` format:

```
x1, y1 ─────────┐
│                │
│   Bounding Box │
│                │
└───────────── x2, y2
```

### 4.2 YOLO Label Format

YOLOv8 requires a `.txt` label file with the same name as each image. Each line represents one object:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates and dimensions are normalized to [0, 1]:

```
x_center = (x1 + x2) / 2 / image_width
y_center = (y1 + y2) / 2 / image_height
width    = (x2 - x1) / image_width
height   = (y2 - y1) / image_height
```

### 4.3 Class ID Mapping

```python
CLASS_MAP = {
    'pedestrian':    0,
    'rider':         1,
    'car':           2,
    'truck':         3,
    'bus':           4,
    'train':         5,
    'motorcycle':    6,
    'bicycle':       7,
    'traffic light':  8,
    'traffic sign':   9,
}
```

### 4.4 Conversion Script

The following script converts BDD100K JSON annotations to YOLO format txt files:

```python
"""
bdd2yolo.py — Convert BDD100K JSON labels to YOLO txt format
Usage:
    python tools/bdd2yolo.py --json-path datasets/bdd100k/labels/bdd100k_labels_images_train.json \
                             --output-dir datasets/bdd100k_weather/labels/train/ \
                             --img-width 1280 --img-height 720
"""

import json
import os
import argparse

CLASS_MAP = {
    'pedestrian': 0, 'rider': 1, 'car': 2, 'truck': 3, 'bus': 4,
    'train': 5, 'motorcycle': 6, 'bicycle': 7,
    'traffic light': 8, 'traffic sign': 9,
}


def convert_bdd_to_yolo(json_path, output_dir, img_width=1280, img_height=720):
    """Convert BDD100K JSON annotations to YOLO txt format"""
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)

    converted = 0
    skipped = 0

    for item in data:
        img_name = item['name']
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_name)

        labels = item.get('labels', [])
        lines = []

        for label in labels:
            category = label.get('category', '')
            if category not in CLASS_MAP:
                continue

            box = label.get('box2d', None)
            if box is None:
                continue

            x1, y1 = box['x1'], box['y1']
            x2, y2 = box['x2'], box['y2']

            # Boundary check
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(0, min(x2, img_width))
            y2 = max(0, min(y2, img_height))

            # Skip invalid boxes (area too small)
            if (x2 - x1) < 1 or (y2 - y1) < 1:
                skipped += 1
                continue

            # Convert to YOLO normalized format
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height

            class_id = CLASS_MAP[category]
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Write empty file even if no objects (YOLO requires it)
        with open(txt_path, 'w') as f:
            f.write('\n'.join(lines))

        converted += 1

    print(f"[INFO] Conversion complete: {converted} images, skipped invalid boxes: {skipped}")
    print(f"[INFO] Output directory: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BDD100K JSON → YOLO txt')
    parser.add_argument('--json-path', type=str, required=True, help='BDD100K JSON annotation file path')
    parser.add_argument('--output-dir', type=str, required=True, help='YOLO txt output directory')
    parser.add_argument('--img-width', type=int, default=1280, help='Image width (default: 1280)')
    parser.add_argument('--img-height', type=int, default=720, help='Image height (default: 720)')
    args = parser.parse_args()
    convert_bdd_to_yolo(args.json_path, args.output_dir, args.img_width, args.img_height)
```

---

## 5. Weather Label Extraction

The `attributes.weather` field in BDD100K contains 6 weather labels. This project maps them into 5 categories:

| BDD100K Original Label | Project Category | Notes |
|----------------------|-----------------|-------|
| clear | clear | Clear sky |
| partly cloudy | clear | Few clouds, merged into clear |
| overcast | overcast | Overcast sky |
| rainy | rainy | Rainy conditions |
| snowy | snowy | Snowy conditions |
| foggy | foggy | Foggy conditions |

The following script extracts weather labels from the JSON and groups images by weather category:

```python
"""
extract_weather.py — Extract weather labels from BDD100K JSON and group images by category
Usage:
    python tools/extract_weather.py --json-path datasets/bdd100k/labels/bdd100k_labels_images_train.json
"""

import json
import argparse
from collections import defaultdict

# BDD100K original weather labels → project 5-class mapping
WEATHER_MAP = {
    'clear':         'clear',
    'partly cloudy': 'clear',      # Few clouds, merged into clear
    'overcast':      'overcast',
    'rainy':         'rainy',
    'snowy':         'snowy',
    'foggy':         'foggy',
}


def extract_weather_groups(json_path):
    """Group images by weather category, return {weather: [image_name, ...]}"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    groups = defaultdict(list)
    unmapped = defaultdict(int)

    for item in data:
        weather_raw = item.get('attributes', {}).get('weather', 'unknown')
        weather = WEATHER_MAP.get(weather_raw, None)

        if weather is None:
            unmapped[weather_raw] += 1
            continue

        groups[weather].append(item['name'])

    # Print statistics
    print(f"Weather distribution (total: {sum(len(v) for v in groups.values())} images):")
    print("-" * 40)
    for w in ['clear', 'rainy', 'foggy', 'snowy', 'overcast']:
        count = len(groups.get(w, []))
        print(f"  {w:<12} {count:>6} images")

    if unmapped:
        print(f"\nUnmapped weather labels:")
        for k, v in unmapped.items():
            print(f"  {k}: {v} images")

    return groups


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-path', type=str, required=True)
    args = parser.parse_args()
    extract_weather_groups(args.json_path)
```

---

## 6. Weather-Balanced Sampling

The sample counts across weather categories in BDD100K are highly imbalanced (clear weather images vastly outnumber foggy or snowy ones). Training directly on the raw dataset would cause the model to underfit on minority classes. Weather-balanced sampling ensures roughly equal representation of each weather category.

### 6.1 Sampling Strategy

- Use the minimum class count N_min as the baseline
- Sample N_min images from each weather category (if a category has fewer than N_min images, use all available)
- Set a random seed for reproducibility
- The final dataset contains approximately 15k images (5 categories × ~3k images/category)

### 6.2 Sampling Script

```python
"""
weather_balanced_sample.py — Weather-balanced sampling from BDD100K
Usage:
    python tools/weather_balanced_sample.py \
        --json-path datasets/bdd100k/labels/bdd100k_labels_images_train.json \
        --image-dir datasets/bdd100k/images/100k/train/ \
        --label-dir datasets/bdd100k_yolo/labels/train/ \
        --output-dir datasets/bdd100k_weather/ \
        --samples-per-weather 3000
"""

import json
import os
import shutil
import random
import argparse
from collections import defaultdict

WEATHER_MAP = {
    'clear': 'clear', 'partly cloudy': 'clear',
    'overcast': 'overcast', 'rainy': 'rainy',
    'snowy': 'snowy', 'foggy': 'foggy',
}


def balanced_sample(json_path, image_dir, label_dir, output_dir,
                    samples_per_weather=3000, seed=42):
    """Weather-balanced sampling, copy images and labels to output directory"""

    # 1. Group by weather
    with open(json_path, 'r') as f:
        data = json.load(f)

    groups = defaultdict(list)
    for item in data:
        weather_raw = item.get('attributes', {}).get('weather', 'unknown')
        weather = WEATHER_MAP.get(weather_raw, None)
        if weather:
            groups[weather].append(item['name'])

    # 2. Balanced sampling
    random.seed(seed)
    sampled = {}
    for weather, images in groups.items():
        n = min(samples_per_weather, len(images))
        sampled[weather] = random.sample(images, n)
        print(f"  {weather:<12}: {len(images):>5} available → sampled {n}")

    # 3. Create output directories
    out_img_dir = os.path.join(output_dir, 'images', 'train')
    out_lbl_dir = os.path.join(output_dir, 'labels', 'train')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    # 4. Copy files
    total = 0
    missing_img = 0
    missing_lbl = 0

    for weather, images in sampled.items():
        for img_name in images:
            # Copy image
            src_img = os.path.join(image_dir, img_name)
            if not os.path.exists(src_img):
                missing_img += 1
                continue
            shutil.copy2(src_img, os.path.join(out_img_dir, img_name))

            # Copy label
            lbl_name = os.path.splitext(img_name)[0] + '.txt'
            src_lbl = os.path.join(label_dir, lbl_name)
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, os.path.join(out_lbl_dir, lbl_name))
            else:
                # Create empty label file
                open(os.path.join(out_lbl_dir, lbl_name), 'w').close()
                missing_lbl += 1

            total += 1

    print(f"\n[INFO] Sampling complete:")
    print(f"  Total copied: {total} images")
    print(f"  Missing images: {missing_img}")
    print(f"  Missing labels: {missing_lbl} (empty files created)")
    print(f"  Output directory: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BDD100K weather-balanced sampling')
    parser.add_argument('--json-path', type=str, required=True,
                        help='BDD100K training JSON annotation path')
    parser.add_argument('--image-dir', type=str, required=True,
                        help='BDD100K original image directory (images/100k/train)')
    parser.add_argument('--label-dir', type=str, required=True,
                        help='Converted YOLO label directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for balanced subset')
    parser.add_argument('--samples-per-weather', type=int, default=3000,
                        help='Number of samples per weather category (default: 3000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()
    balanced_sample(args.json_path, args.image_dir, args.label_dir,
                    args.output_dir, args.samples_per_weather, args.seed)
```

---

## 7. Final Directory Structure

After completing all the steps above, the dataset directory structure should look like this:

```
datasets/
├── bdd100k/                        # Raw data (optional, can be kept or removed)
│   ├── images/100k/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── bdd100k_labels_images_train.json
│       └── bdd100k_labels_images_val.json
│
└── bdd100k_weather/                # Weather-balanced subset (used for training)
    ├── images/
    │   ├── train/                  # ~15,000 weather-balanced images
    │   └── val/                    # Validation images
    └── labels/
        ├── train/                  # Corresponding YOLO txt labels
        └── val/                    # Validation labels
```

The validation set follows the same processing pipeline (JSON → YOLO txt conversion), but is typically not subject to balanced sampling. Keeping the original distribution in the validation set better reflects real-world performance.

---

## 8. Dataset Configuration File

YOLOv8 requires a dataset configuration yaml file for training, located at `configs/data/bdd100k_weather.yaml`:

```yaml
# BDD100K Weather-Balanced Dataset Configuration
# Modify 'path' to your local dataset path before use

path: ./datasets/bdd100k_weather   # Dataset root directory (change to your actual path)
train: images/train                # Training image directory (relative to path)
val: images/val                    # Validation image directory (relative to path)

# Class information (BDD100K 10 classes)
nc: 10
names:
  - pedestrian
  - rider
  - car
  - truck
  - bus
  - train
  - motorcycle
  - bicycle
  - traffic light
  - traffic sign
```

---

## 9. FAQ

### Q: What if the download speed is too slow?

The BDD100K official server is hosted overseas. You can try Kaggle mirrors or use multi-threaded download tools such as aria2 for faster downloads.

### Q: Some images have no annotated objects after label conversion. Is this normal?

Yes, this is expected. Some images in BDD100K genuinely contain no annotated objects (e.g., empty highway scenes). YOLO handles empty label files automatically during training — these images serve as negative samples, which help reduce false positives.

### Q: What if foggy and snowy categories have too few samples?

Foggy and snowy images are indeed scarce in BDD100K. If a category has fewer images than the target sampling count, the script will automatically use all available samples. Moderate differences in sample counts (within ±30%) have minimal impact on training. For larger imbalances, consider applying data augmentation (e.g., Mosaic, MixUp) to compensate for underrepresented categories.

### Q: Should the validation set also be weather-balanced?

No. The validation set should retain its original distribution so that evaluation results reflect real-world model performance. Weather-balanced sampling should only be applied to the training set.

### Q: Why is "partly cloudy" merged into "clear" instead of "overcast"?

"Partly cloudy" (few clouds) has visual characteristics more similar to clear weather: sufficient lighting, high contrast, and clearly visible objects. In contrast, "overcast" typically involves overall dimness and low contrast. From an object detection perspective, the impact of "partly cloudy" on detection performance is more similar to "clear" than to "overcast".