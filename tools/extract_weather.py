"""
extract_weather.py — 从 BDD100K JSON 中提取天气标签

按天气类别分组统计图像数量，用于了解数据分布和规划采样策略。

用法:
    python tools/extract_weather.py \
        --json-path datasets/bdd100k/labels/bdd100k_labels_images_train.json
"""

import json
import argparse
from collections import defaultdict

# BDD100K 原始天气标签 → 本项目 5 类映射
WEATHER_MAP = {
    'clear':         'clear',
    'partly cloudy': 'clear',      # 少云归入晴天
    'overcast':      'overcast',
    'rainy':         'rainy',
    'snowy':         'snowy',
    'foggy':         'foggy',
}

WEATHER_CLASSES = ['clear', 'rainy', 'foggy', 'snowy', 'overcast']


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
    total = sum(len(v) for v in groups.values())
    print(f"天气分布统计 (共 {total} 张):")
    print("-" * 40)
    for w in WEATHER_CLASSES:
        count = len(groups.get(w, []))
        pct = count / total * 100 if total > 0 else 0
        print(f"  {w:<12} {count:>6} 张  ({pct:>5.1f}%)")

    if unmapped:
        print(f"\n未映射的天气标签:")
        for k, v in unmapped.items():
            print(f"  {k}: {v} 张")

    return groups


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BDD100K 天气标签提取与统计')
    parser.add_argument('--json-path', type=str, required=True,
                        help='BDD100K JSON 标注文件路径')
    args = parser.parse_args()
    extract_weather_groups(args.json_path)