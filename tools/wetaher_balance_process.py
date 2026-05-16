"""
weather_balanced_sample.py — BDD100K 天气均衡采样

从 BDD100K 中按天气条件均衡采样，生成训练用子集。
以各类别中最少样本数为基准（或指定上限），确保各天气类别样本量大致相等。

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

WEATHER_CLASSES = ['clear', 'rainy', 'foggy', 'snowy', 'overcast']


def balanced_sample(json_path, image_dir, label_dir, output_dir,
                    samples_per_weather=3000, seed=42):
    """天气均衡采样，复制图像和标签到输出目录"""

    # 1. 读取 JSON，按天气分组
    with open(json_path, 'r') as f:
        data = json.load(f)

    groups = defaultdict(list)
    for item in data:
        weather_raw = item.get('attributes', {}).get('weather', 'unknown')
        weather = WEATHER_MAP.get(weather_raw, None)
        if weather:
            groups[weather].append(item['name'])

    # 2. 均衡采样
    print("天气均衡采样:")
    print("-" * 50)
    random.seed(seed)
    sampled = {}
    for weather in WEATHER_CLASSES:
        images = groups.get(weather, [])
        n = min(samples_per_weather, len(images))
        sampled[weather] = random.sample(images, n) if images else []
        print(f"  {weather:<12}: {len(images):>5} 张可用 → 采样 {n} 张")

    total_sampled = sum(len(v) for v in sampled.values())
    print(f"  {'合计':<12}: 采样 {total_sampled} 张")

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
                        help='BDD100K 训练集 JSON 标注路径')
    parser.add_argument('--image-dir', type=str, required=True,
                        help='BDD100K 原始图像目录 (images/100k/train)')
    parser.add_argument('--label-dir', type=str, required=True,
                        help='已转换的 YOLO 标签目录')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='均衡采样输出目录')
    parser.add_argument('--samples-per-weather', type=int, default=3000,
                        help='每类天气采样数量 (默认 3000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认 42)')
    args = parser.parse_args()
    balanced_sample(args.json_path, args.image_dir, args.label_dir,
                    args.output_dir, args.samples_per_weather, args.seed)