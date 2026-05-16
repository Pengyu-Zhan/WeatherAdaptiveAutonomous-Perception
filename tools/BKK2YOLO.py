"""
bdd2yolo.py — BDD100K JSON 标签转 YOLO txt 格式

将 BDD100K 的 JSON 标注文件转换为 YOLOv8 所需的 txt 标签格式。
每张图像生成一个同名 .txt 文件，每行格式:
    <class_id> <x_center> <y_center> <width> <height>
所有坐标归一化到 [0, 1]。

用法:
    python tools/bdd2yolo.py \
        --json-path datasets/bdd100k/labels/bdd100k_labels_images_train.json \
        --output-dir datasets/bdd100k_yolo/labels/train/ \
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
    skipped_boxes = 0

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
                skipped_boxes += 1
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

    print(f"[INFO] 转换完成: {converted} 张, 跳过无效框: {skipped_boxes} 个")
    print(f"[INFO] 输出目录: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BDD100K JSON → YOLO txt')
    parser.add_argument('--json-path', type=str, required=True,
                        help='BDD100K JSON 标注文件路径')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='YOLO txt 输出目录')
    parser.add_argument('--img-width', type=int, default=1280,
                        help='图像宽度 (默认 1280)')
    parser.add_argument('--img-height', type=int, default=720,
                        help='图像高度 (默认 720)')
    args = parser.parse_args()
    convert_bdd_to_yolo(args.json_path, args.output_dir, args.img_width, args.img_height)