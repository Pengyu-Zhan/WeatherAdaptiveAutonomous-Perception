"""
ONNX Model Export
=================
Export trained WeatherAdaptive-YOLOv8 model to ONNX format.
Handles EnvAdaptiveFusion module's _original_img_cache requirement.

Output: input [1, 3, 640, 640] → output (1, 14, 8400)

Usage:
    python scripts/export_onnx.py --weights runs/train/best.pt --imgsz 640

"""

import argparse
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument('--weights', type=str, required=True, help='Model weights path (.pt)')
    parser.add_argument('--imgsz', type=int, default=640, help='Export image size')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', default=True, help='Simplify ONNX model')
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.weights)

    # EnvAdaptiveFusion 模块在 ONNX trace 时需要 _original_img_cache
    # 塞一个 dummy 输入使 trace 走正常的天气推理路径
    dummy_img = torch.randn(1, 3, args.imgsz, args.imgsz)

    for name, module in model.model.named_modules():
        if module.__class__.__name__ == "EnvAdaptiveFusion":
            module.__class__._original_img_cache = dummy_img
            print(f"[INFO] Set _original_img_cache for module: {name}")
            break

    model.export(format="onnx", opset=args.opset, simplify=args.simplify, imgsz=args.imgsz)
    print("[INFO] ONNX export completed.")


if __name__ == '__main__':
    main()