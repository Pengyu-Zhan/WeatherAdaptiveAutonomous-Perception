# 数据准备指南

[zh_CN 中文](data_preparation_cn.md)

本文档详细介绍如何从零开始准备 BDD100K 天气均衡数据集，用于 WeatherAdaptive-Perception 项目的训练与评估。

---

## 目录

- [1. BDD100K 数据集简介](#1-bdd100k-数据集简介)
- [2. 数据集下载](#2-数据集下载)
- [3. 原始数据结构](#3-原始数据结构)
- [4. 标签格式说明与转换](#4-标签格式说明与转换)
- [5. 天气标签提取](#5-天气标签提取)
- [6. 天气均衡采样](#6-天气均衡采样)
- [7. 最终目录结构](#7-最终目录结构)
- [8. 数据集配置文件](#8-数据集配置文件)
- [9. 常见问题](#9-常见问题)

---

## 1. BDD100K 数据集简介

[BDD100K](https://www.bdd100k.com/) 是加州大学伯克利分校发布的大规模自动驾驶数据集，包含 10 万张驾驶场景图像，涵盖多种天气、时间、场景条件，适合用于研究环境自适应的目标检测模型。

数据集的关键属性标注：

| 属性类别 | 可选值 |
|---------|--------|
| 天气 (weather) | clear, partly cloudy, overcast, rainy, snowy, foggy |
| 时段 (timeofday) | daytime, night, dawn/dusk |
| 场景 (scene) | city street, highway, residential, parking lot, gas station, tunnel |

目标检测标注共 10 个类别：pedestrian, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign。

本项目从 BDD100K 中按天气条件均衡采样约 15k 张图像，构建天气均衡子集用于训练。

---

## 2. 数据集下载

### 2.1 官方渠道

访问 [BDD100K 官网](https://www.bdd100k.com/)，注册账号后下载以下文件：

- **100K Images** (`bdd100k_images_100k.zip`, ~6.5GB) — 全部图像
- **Detection Labels** (`bdd100k_labels_release.zip`, ~100MB) — 检测标注 JSON

### 2.2 Kaggle（备选）

如果官网下载速度慢，也可以在 Kaggle 上搜索 BDD100K 数据集，部分用户已上传镜像。

### 2.3 下载完成后

将两个 zip 文件解压到同一个根目录下：

```bash
mkdir -p datasets/bdd100k
cd datasets/bdd100k

# 解压图像
unzip bdd100k_images_100k.zip

# 解压标签
unzip bdd100k_labels_release.zip
```

---

## 3. 原始数据结构

解压后的目录结构如下：

```
datasets/bdd100k/
├── images/
│   └── 100k/
│       ├── train/          # 70,000 张训练图像 (1280×720, .jpg)
│       ├── val/            # 10,000 张验证图像
│       └── test/           # 20,000 张测试图像（无标签）
└── labels/
    └── bdd100k_labels_images_train.json    # 训练集标注
    └── bdd100k_labels_images_val.json      # 验证集标注
```

### 3.1 标签 JSON 结构

每个 JSON 文件包含一个列表，每个元素对应一张图像：

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

其中 `attributes.weather` 是我们进行天气均衡采样的关键字段。

---

## 4. 标签格式说明与转换

### 4.1 BDD100K 原始格式

BDD100K 使用绝对像素坐标的 `(x1, y1, x2, y2)` 格式：

```
x1, y1 ─────────┐
│                │
│    目标框       │
│                │
└───────────── x2, y2
```

### 4.2 YOLO 标签格式

YOLOv8 要求每张图像对应一个同名 `.txt` 标签文件，每行一个目标：

```
<class_id> <x_center> <y_center> <width> <height>
```

所有坐标和尺寸均为相对值（归一化到 0~1）：

```
x_center = (x1 + x2) / 2 / image_width
y_center = (y1 + y2) / 2 / image_height
width    = (x2 - x1) / image_width
height   = (y2 - y1) / image_height
```

### 4.3 类别 ID 映射

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

### 4.4 转换脚本

以下脚本将 BDD100K JSON 标注转换为 YOLO 格式的 txt 文件：

```python
"""
bdd2yolo.py — BDD100K JSON 标签转 YOLO txt 格式
用法:
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
    """将 BDD100K JSON 标注转为 YOLO txt 格式"""
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

            # 边界检查
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(0, min(x2, img_width))
            y2 = max(0, min(y2, img_height))

            # 跳过无效框（面积过小）
            if (x2 - x1) < 1 or (y2 - y1) < 1:
                skipped += 1
                continue

            # 转换为 YOLO 归一化格式
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height

            class_id = CLASS_MAP[category]
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # 即使没有目标也写空文件（YOLO 需要）
        with open(txt_path, 'w') as f:
            f.write('\n'.join(lines))

        converted += 1

    print(f"[INFO] 转换完成: {converted} 张, 跳过无效框: {skipped} 个")
    print(f"[INFO] 输出目录: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BDD100K JSON → YOLO txt')
    parser.add_argument('--json-path', type=str, required=True, help='BDD100K JSON 标注文件路径')
    parser.add_argument('--output-dir', type=str, required=True, help='YOLO txt 输出目录')
    parser.add_argument('--img-width', type=int, default=1280, help='图像宽度 (默认 1280)')
    parser.add_argument('--img-height', type=int, default=720, help='图像高度 (默认 720)')
    args = parser.parse_args()
    convert_bdd_to_yolo(args.json_path, args.output_dir, args.img_width, args.img_height)
```

---

## 5. 天气标签提取

BDD100K 的 `attributes.weather` 字段包含 6 种天气标签。本项目将其映射为 5 类：

| BDD100K 原始标签 | 本项目类别 | 说明 |
|-----------------|-----------|------|
| clear           | clear     | 晴天 |
| partly cloudy   | clear     | 少云，归入晴天 |
| overcast        | overcast  | 阴天 |
| rainy           | rainy     | 雨天 |
| snowy           | snowy     | 雪天 |
| foggy           | foggy     | 雾天 |

以下脚本从 JSON 中提取每张图像的天气标签，并按天气类别分组：

```python
"""
extract_weather.py — 从 BDD100K JSON 中提取天气标签，按天气分组图像列表
用法:
    python tools/extract_weather.py --json-path datasets/bdd100k/labels/bdd100k_labels_images_train.json
"""

import json
import argparse
from collections import defaultdict

# BDD100K 原始天气标签 → 本项目 5 类映射
WEATHER_MAP = {
    'clear':         'clear',
    'partly cloudy': 'clear',     # 少云归入晴天
    'overcast':      'overcast',
    'rainy':         'rainy',
    'snowy':         'snowy',
    'foggy':         'foggy',
}


def extract_weather_groups(json_path):
    """按天气类别分组，返回 {weather: [image_name, ...]}"""
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

    # 打印统计
    print(f"天气分布统计 (共 {sum(len(v) for v in groups.values())} 张):")
    print("-" * 40)
    for w in ['clear', 'rainy', 'foggy', 'snowy', 'overcast']:
        count = len(groups.get(w, []))
        print(f"  {w:<12} {count:>6} 张")

    if unmapped:
        print(f"\n未映射的天气标签:")
        for k, v in unmapped.items():
            print(f"  {k}: {v} 张")

    return groups


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-path', type=str, required=True)
    args = parser.parse_args()
    extract_weather_groups(args.json_path)
```

---

## 6. 天气均衡采样

BDD100K 中各天气类别的样本数量极不均衡（晴天远多于雾天、雪天），直接训练会导致模型对少数类欠拟合。因此需要进行均衡采样，使每种天气类别的样本数量大致相等。

### 6.1 采样策略

- 以最少类别的样本数 N_min 为基准
- 每种天气类别取 N_min 张（如果该类别不足 N_min 张则全部取用）
- 设置随机种子保证可复现
- 最终数据集约 15k 张（5 类 × ~3k 张/类）

### 6.2 采样脚本

```python
"""
weather_balanced_sample.py — 天气均衡采样
从 BDD100K 中按天气均衡采样，生成训练用子集。

用法:
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
    """天气均衡采样，复制图像和标签到输出目录"""

    # 1. 按天气分组
    with open(json_path, 'r') as f:
        data = json.load(f)

    groups = defaultdict(list)
    for item in data:
        weather_raw = item.get('attributes', {}).get('weather', 'unknown')
        weather = WEATHER_MAP.get(weather_raw, None)
        if weather:
            groups[weather].append(item['name'])

    # 2. 均衡采样
    random.seed(seed)
    sampled = {}
    for weather, images in groups.items():
        n = min(samples_per_weather, len(images))
        sampled[weather] = random.sample(images, n)
        print(f"  {weather:<12}: {len(images):>5} 张可用 → 采样 {n} 张")

    # 3. 创建输出目录
    out_img_dir = os.path.join(output_dir, 'images', 'train')
    out_lbl_dir = os.path.join(output_dir, 'labels', 'train')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    # 4. 复制文件
    total = 0
    missing_img = 0
    missing_lbl = 0

    for weather, images in sampled.items():
        for img_name in images:
            # 复制图像
            src_img = os.path.join(image_dir, img_name)
            if not os.path.exists(src_img):
                missing_img += 1
                continue
            shutil.copy2(src_img, os.path.join(out_img_dir, img_name))

            # 复制标签
            lbl_name = os.path.splitext(img_name)[0] + '.txt'
            src_lbl = os.path.join(label_dir, lbl_name)
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, os.path.join(out_lbl_dir, lbl_name))
            else:
                # 创建空标签文件
                open(os.path.join(out_lbl_dir, lbl_name), 'w').close()
                missing_lbl += 1

            total += 1

    print(f"\n[INFO] 采样完成:")
    print(f"  总计复制: {total} 张")
    print(f"  图像缺失: {missing_img} 张")
    print(f"  标签缺失: {missing_lbl} 张（已创建空文件）")
    print(f"  输出目录: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BDD100K 天气均衡采样')
    parser.add_argument('--json-path', type=str, required=True,
                        help='BDD100K 训练集 JSON 标注')
    parser.add_argument('--image-dir', type=str, required=True,
                        help='BDD100K 原始图像目录 (images/100k/train)')
    parser.add_argument('--label-dir', type=str, required=True,
                        help='已转换的 YOLO 标签目录')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--samples-per-weather', type=int, default=3000,
                        help='每类天气采样数量 (默认 3000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认 42)')
    args = parser.parse_args()
    balanced_sample(args.json_path, args.image_dir, args.label_dir,
                    args.output_dir, args.samples_per_weather, args.seed)
```

---

## 7. 最终目录结构

完成上述所有步骤后，数据集的目录结构应如下：

```
datasets/
├── bdd100k/                        # 原始数据（可选保留）
│   ├── images/100k/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── bdd100k_labels_images_train.json
│       └── bdd100k_labels_images_val.json
│
└── bdd100k_weather/                # 天气均衡子集（实际训练用）
    ├── images/
    │   ├── train/                  # ~15,000 张天气均衡图像
    │   └── val/                    # 验证集图像
    └── labels/
        ├── train/                  # 对应的 YOLO txt 标签
        └── val/                    # 验证集标签
```

验证集的处理方式与训练集相同（JSON → YOLO txt 转换），但通常不做均衡采样，保持原始分布以反映真实场景。

---

## 8. 数据集配置文件

训练时 YOLOv8 需要一个数据集配置 yaml 文件，位于 `configs/data/bdd100k_weather.yaml`：

```yaml
# BDD100K Weather-Balanced Dataset Configuration
# 使用前请将 path 修改为你本地的数据集实际路径

path: ./datasets/bdd100k_weather   # 数据集根目录（修改为你的实际路径）
train: images/train                # 训练图像目录（相对于 path）
val: images/val                    # 验证图像目录（相对于 path）

# 类别信息（BDD100K 10类）
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

## 9. 常见问题

### Q: 下载速度太慢怎么办？

BDD100K 官网服务器在国外，国内下载可能很慢。可以尝试 Kaggle 镜像，或者使用下载工具（如 aria2）多线程下载。

### Q: 标签转换后发现有些图片没有目标怎么办？

这是正常的。BDD100K 中部分图像确实没有标注目标（比如空旷的高速公路场景）。YOLO 训练时会自动处理空标签文件，这些图像作为负样本参与训练，有助于减少误检。

### Q: foggy 和 snowy 类别样本太少怎么办？

BDD100K 中雾天和雪天图像确实较少。如果某类不足目标采样数量，脚本会自动取用该类全部样本。样本量差异在合理范围内（±30%）对训练影响不大。如果差异过大，可以考虑对少数类做数据增强（如 Mosaic、MixUp）来补偿。

### Q: 验证集需要做天气均衡采样吗？

建议不做。验证集应保持原始分布，这样评估结果才能反映模型在真实场景下的表现。天气均衡只在训练集上做。

### Q: partly cloudy 为什么归入 clear 而不是 overcast？

partly cloudy（少云）的视觉特征更接近晴天：光照充足、对比度高、目标清晰。而 overcast（阴天）通常伴随整体偏暗、低对比度。从目标检测模型的角度看，partly cloudy 对检测的影响与 clear 更相似。