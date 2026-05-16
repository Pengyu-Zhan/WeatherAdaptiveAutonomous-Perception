"""
门控诊断脚本
===========
不依赖 BDD JSON，直接用模型内置的天气分类器判断天气类型，
检查各天气条件下 P3/P4/P5 gate 值是否有分化。

用法:
    python scripts/gate_diagnose.py --weights runs/train/best.pt \
                                    --val-dir datasets/bdd100k_weather/images/val \
                                    --num-samples 200
"""

import argparse
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.nn.modules.block import EnvAdaptiveFusion

# 天气类别（与训练时一致，5 类）
WEATHER_NAMES = ['clear', 'rainy', 'foggy', 'snowy', 'overcast']


def parse_args():
    parser = argparse.ArgumentParser(description='门控诊断：天气模块是否学到差异化映射')
    parser.add_argument('--weights', type=str, required=True,
                        help='模型权重路径 (.pt)')
    parser.add_argument('--val-dir', type=str, required=True,
                        help='验证集图像目录')
    parser.add_argument('--num-samples', type=int, default=200,
                        help='采样图像数量 (默认 200)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='模型输入尺寸 (默认 640)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认 42)')
    parser.add_argument('--top-k', type=int, default=5,
                        help='每类展示前 k 张详细数据 (默认 5)')
    return parser.parse_args()


def load_image(img_path, device, imgsz=640):
    """读取图片，返回 [1, 3, imgsz, imgsz] tensor"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (imgsz, imgsz))
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return tensor.to(device)


def get_env_module(model):
    """从模型中找到 EnvAdaptiveFusion 层"""
    for m in model.model.modules():
        if isinstance(m, EnvAdaptiveFusion):
            return m
    raise RuntimeError("模型中没有 EnvAdaptiveFusion 层！")


def analyze_single_image(env_module, img_tensor):
    """对单张图推理，返回天气概率、gate 值、置信度"""
    device = img_tensor.device

    # 天气分类器输入预处理 (224x224, ImageNet 归一化)
    img_224 = F.interpolate(img_tensor.float(), size=(224, 224),
                            mode='bilinear', align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    img_224 = (img_224 - mean) / std
    img_224 = img_224.to(next(env_module.env_module.parameters()).dtype)

    with torch.no_grad():
        env_module.env_module.eval()
        v = env_module.env_module(img_224).detach()

    weather_probs = v[0].cpu().numpy()
    max_prob = float(v.max())

    # 门控值计算
    v_mapped = v.to(next(env_module.gate_mapper.parameters()).dtype)
    with torch.no_grad():
        gate_logits = env_module.gate_mapper(v_mapped)
        raw_gates = 1.0 + env_module.alpha * torch.tanh(gate_logits)

        is_confident = float(max_prob >= env_module.confidence_threshold)
        gates = raw_gates if is_confident else torch.ones_like(raw_gates)

    return {
        'weather_probs': weather_probs,
        'predicted_weather': WEATHER_NAMES[np.argmax(weather_probs)],
        'confidence': max_prob,
        'is_confident': is_confident,
        'p3_gate': gates[0, 0].item(),
        'p4_gate': gates[0, 1].item(),
        'p5_gate': gates[0, 2].item(),
    }


def main():
    args = parse_args()

    print("=" * 65)
    print("门控诊断：天气模块是否学到差异化映射")
    print("=" * 65)

    # 加载模型
    print("\n[1] 加载模型...")
    model = YOLO(args.weights)
    device = next(model.model.parameters()).device
    env_module = get_env_module(model)
    print(f"    alpha = {env_module.alpha}")
    print(f"    confidence_threshold = {env_module.confidence_threshold}")

    # 扫描图像
    print(f"\n[2] 扫描 {args.val_dir}...")
    all_imgs = [os.path.join(args.val_dir, f) for f in os.listdir(args.val_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    random.seed(args.seed)
    random.shuffle(all_imgs)
    test_imgs = all_imgs[:args.num_samples]
    print(f"    共 {len(all_imgs)} 张，取 {len(test_imgs)} 张分析")

    # 推理
    print(f"\n[3] 推理中...")
    results = {w: [] for w in WEATHER_NAMES}

    for i, path in enumerate(test_imgs):
        img_tensor = load_image(path, device, args.imgsz)
        r = analyze_single_image(env_module, img_tensor)
        weather = r['predicted_weather']
        results[weather].append(r)

        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(test_imgs)} done...")

    # ==================== 诊断结果 ====================
    print("\n" + "=" * 65)
    print("诊断结果")
    print("=" * 65)

    # 天气分布
    print(f"\n天气分布（模型自己的分类结果）:")
    for w in WEATHER_NAMES:
        print(f"  {w}: {len(results[w])} 张")

    # 汇总表
    num_classes = len(WEATHER_NAMES)
    prob_header = "平均概率 " + str([w[:3] for w in WEATHER_NAMES])
    print(f"\n{'天气':<10} {'数量':>5} {'P3 gate':>10} {'P4 gate':>10} {'P5 gate':>10} "
          f"{'置信度':>8}  {prob_header}")
    print("-" * 95)

    gate_avgs = {}
    for w in WEATHER_NAMES:
        data = results[w]
        if not data:
            print(f"{w:<10} {'0':>5}     (无数据)")
            continue

        n = len(data)
        p3 = np.mean([d['p3_gate'] for d in data])
        p4 = np.mean([d['p4_gate'] for d in data])
        p5 = np.mean([d['p5_gate'] for d in data])
        conf = np.mean([d['confidence'] for d in data])
        probs = np.mean([d['weather_probs'] for d in data], axis=0)

        gate_avgs[w] = {'p3': p3, 'p4': p4, 'p5': p5}
        probs_str = ', '.join([f'{p:.3f}' for p in probs])
        print(f"{w:<10} {n:>5} {p3:>10.6f} {p4:>10.6f} {p5:>10.6f} "
              f"{conf:>8.3f}   [{probs_str}]")

    # 差异分析
    print(f"\n差异分析:")
    if len(gate_avgs) >= 2:
        for gate_name in ['p3', 'p4', 'p5']:
            vals = {w: gate_avgs[w][gate_name] for w in gate_avgs}
            spread = max(vals.values()) - min(vals.values())
            best_w = max(vals, key=vals.get)
            worst_w = min(vals, key=vals.get)
            print(f"  {gate_name.upper()}: 极差={spread:.6f}  "
                  f"最高={best_w}({vals[best_w]:.6f})  "
                  f"最低={worst_w}({vals[worst_w]:.6f})")

        all_spreads = []
        for gate_name in ['p3', 'p4', 'p5']:
            vals = list({w: gate_avgs[w][gate_name] for w in gate_avgs}.values())
            all_spreads.append(max(vals) - min(vals))
        avg_spread = np.mean(all_spreads)

        print(f"\n  平均极差: {avg_spread:.6f}")
        if avg_spread < 0.005:
            print("  ⚠️  gate 几乎无分化，模块等于恒等映射。")
        elif avg_spread < 0.02:
            print("  🔶 gate 有微弱分化，幅度太小，对检测影响有限。")
        else:
            print("  ✅ gate 有明显分化，天气模块在差异化调节。")

    # 每类详细数据
    print(f"\n" + "=" * 65)
    print(f"每类前 {args.top_k} 张详细数据")
    print("=" * 65)
    for w in WEATHER_NAMES:
        print(f"\n--- {w} ---")
        if not results[w]:
            print("  (无数据)")
            continue
        for d in results[w][:args.top_k]:
            p = d['weather_probs']
            probs_str = ', '.join([f'{x:.3f}' for x in p])
            print(f"  P3={d['p3_gate']:.6f}  P4={d['p4_gate']:.6f}  P5={d['p5_gate']:.6f}  "
                  f"conf={d['confidence']:.3f}  probs=[{probs_str}]")


if __name__ == '__main__':
    main()