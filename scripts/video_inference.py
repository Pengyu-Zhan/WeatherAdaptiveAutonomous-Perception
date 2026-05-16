"""
视频推理管线 — 基于天气感知的动态跳连网络道路场景检测

=================================================
实现专利中描述的自适应视频抽帧机制：
  Δn = ⌊f₀ / f_s⌋
非抽取帧复用最近一次检测结果，保持视觉连续性的同时降低计算开销。

用法:
  python scripts/video_inference.py --source driving_video.mp4 --weights best.pt
  python scripts/video_inference.py --source driving_video.mp4 --weights best.engine  # TensorRT
  python scripts/video_inference.py --source 0  # 摄像头实时推理
"""

import argparse
import time
import cv2
import numpy as np
import torch
from pathlib import Path

# ============================================================
#  默认配置参数
# ============================================================

# 专利抽帧参数
DEFAULT_TARGET_FPS = 10       # f_s: 抽帧频率（帧/秒）
DEFAULT_CONF_THRESHOLD = 0.25 # 检测置信度阈值
DEFAULT_IOU_THRESHOLD = 0.45  # NMS IoU 阈值
DEFAULT_INPUT_SIZE = 640      # 模型输入尺寸

# 天气类别映射（与训练时一致，5 类）
WEATHER_NAMES = {0: "clear", 1: "rainy", 2: "foggy", 3: "snowy", 4: "overcast"}
WEATHER_COLORS = {
    "clear":    (0, 200, 255),   # 橙黄
    "rainy":    (255, 150, 0),   # 蓝
    "foggy":    (200, 200, 200), # 灰白
    "snowy":    (255, 230, 180), # 浅蓝
    "overcast": (150, 150, 150), # 灰
}

# 可视化颜色（BGR）
BOX_COLOR = (0, 255, 0)
TEXT_BG_COLOR = (0, 0, 0)
FPS_COLOR = (0, 255, 255)


# ============================================================
#  天气分类提取器（从模型内部钩取天气预测结果）
# ============================================================

class WeatherHook:
    """
    通过 PyTorch forward hook 拦截 EnvAdaptiveFusion 模块内部的
    天气分类器输出，用于在画面上显示当前天气预测结果。

    如果模型中没有找到 EnvAdaptiveFusion 模块（比如跑的是标准
    YOLOv8 baseline），则安静地返回 None，不影响正常推理。
    """

    def __init__(self):
        self.weather_probs = None  # 最近一次的天气概率向量
        self.weather_label = None  # 最近一次的天气类别名称
        self._handle = None

    def register(self, model):
        """在模型中查找 EnvAdaptiveFusion 模块并注册 hook"""
        try:
            torch_model = model.model
            for name, module in torch_model.named_modules():
                if module.__class__.__name__ == "EnvAdaptiveFusion":
                    self._handle = module.register_forward_hook(self._hook_fn)
                    print(f"[WeatherHook] 已挂载到模块: {name}")
                    return True
            print("[WeatherHook] 未找到 EnvAdaptiveFusion 模块，天气显示将不可用")
            return False
        except Exception as e:
            print(f"[WeatherHook] 注册失败: {e}")
            return False

    def _hook_fn(self, module, input, output):
        """hook 回调：从模块中提取天气分类结果"""
        try:
            if hasattr(module, 'weather_probs'):
                probs = module.weather_probs
            elif hasattr(module, 'last_weather_output'):
                probs = module.last_weather_output
            else:
                return

            if probs is not None:
                if isinstance(probs, torch.Tensor):
                    probs = probs.detach().cpu().numpy().flatten()
                self.weather_probs = probs
                weather_idx = int(np.argmax(probs))
                self.weather_label = WEATHER_NAMES.get(weather_idx, f"class_{weather_idx}")
        except Exception:
            pass  # 静默失败，不影响主流程

    def get_weather_info(self):
        """返回当前天气信息，用于可视化"""
        if self.weather_label is None:
            return None, None
        return self.weather_label, self.weather_probs

    def remove(self):
        """移除 hook"""
        if self._handle is not None:
            self._handle.remove()


# ============================================================
#  可视化工具
# ============================================================

def draw_detections(frame, boxes, scores, class_ids, class_names):
    """在帧上绘制检测框和标签"""
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[int(cls_id)]} {score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), TEXT_BG_COLOR, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def draw_hud(frame, fps_infer, fps_overall, weather_label, weather_probs,
             frame_idx, is_detection_frame, total_frames):
    """绘制 HUD 信息面板（左上角）"""
    h, w = frame.shape[:2]

    # 半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (320, 170), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = 32
    line_h = 24
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55

    # 推理 FPS
    color = (0, 255, 0) if fps_infer >= 10 else (0, 165, 255) if fps_infer >= 5 else (0, 0, 255)
    cv2.putText(frame, f"Infer FPS: {fps_infer:.1f}", (20, y), font, font_scale, color, 1)
    y += line_h

    # 整体 FPS
    cv2.putText(frame, f"Total FPS: {fps_overall:.1f}", (20, y), font, font_scale, FPS_COLOR, 1)
    y += line_h

    # 帧信息
    status = "DETECT" if is_detection_frame else "REUSE"
    status_color = (0, 255, 0) if is_detection_frame else (180, 180, 180)
    progress = f"{frame_idx}/{total_frames}" if total_frames > 0 else f"{frame_idx}"
    cv2.putText(frame, f"Frame: {progress}  [{status}]", (20, y),
                font, font_scale, status_color, 1)
    y += line_h

    # 天气信息
    if weather_label is not None:
        w_color = WEATHER_COLORS.get(weather_label, (255, 255, 255))
        cv2.putText(frame, f"Weather: {weather_label}", (20, y), font, font_scale, w_color, 1)
        y += line_h

        # 天气概率条
        if weather_probs is not None:
            bar_x = 20
            bar_w = 55
            for i, (name, prob) in enumerate(zip(WEATHER_NAMES.values(), weather_probs)):
                short = name[:3].upper()
                cv2.putText(frame, f"{short}", (bar_x, y), font, 0.35, (200, 200, 200), 1)
                bx = bar_x + 30
                cv2.rectangle(frame, (bx, y - 10), (bx + bar_w, y), (60, 60, 60), -1)
                fill_w = int(bar_w * prob)
                bar_color = WEATHER_COLORS.get(name, (255, 255, 255))
                cv2.rectangle(frame, (bx, y - 10), (bx + fill_w, y), bar_color, -1)
                bar_x += 60
    else:
        cv2.putText(frame, "Weather: N/A", (20, y), font, font_scale, (128, 128, 128), 1)

    return frame


# ============================================================
#  核心推理管线
# ============================================================

def run_video_inference(args):
    """主推理循环"""

    from ultralytics import YOLO
    print(f"[INFO] 加载模型: {args.weights}")
    model = YOLO(args.weights)
    class_names = model.names

    # 尝试注册天气 hook
    weather_hook = WeatherHook()
    has_weather = weather_hook.register(model)

    # 打开视频源（支持视频文件路径或摄像头编号）
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频源: {args.source}")
        return

    # 获取视频属性
    f0 = cap.get(cv2.CAP_PROP_FPS)
    if f0 <= 0:
        f0 = 30.0
        print(f"[WARN] 无法获取视频帧率，默认 {f0} FPS")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 计算抽帧间隔（专利公式）
    fs = args.target_fps
    delta_n = max(1, int(f0 // fs))  # Δn = ⌊f₀ / f_s⌋

    print(f"[INFO] 视频属性: {frame_w}x{frame_h} @ {f0:.1f} FPS, 共 {total_frames} 帧")
    print(f"[INFO] 抽帧配置: f_s={fs} FPS, Δn={delta_n} (每 {delta_n} 帧检测 1 帧)")
    print(f"[INFO] 计算负载降低: {(1 - 1 / delta_n) * 100:.0f}%")

    # 初始化视频写入器
    writer = None
    if args.output:
        output_path = str(Path(args.output))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, f0, (frame_w, frame_h))
        print(f"[INFO] 输出视频: {output_path}")

    # 推理循环
    frame_idx = 0
    detect_count = 0
    reuse_count = 0

    last_boxes = np.array([])
    last_scores = np.array([])
    last_class_ids = np.array([])
    last_weather_label = None
    last_weather_probs = None

    fps_infer = 0.0
    fps_overall = 0.0
    time_start = time.time()
    infer_times = []

    print(f"[INFO] 开始推理... (按 'q' 退出)")
    print("-" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        is_detection_frame = (frame_idx % delta_n == 1) or (delta_n == 1)

        if is_detection_frame:
            detect_count += 1
            t_infer_start = time.time()

            results = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                verbose=False
            )

            t_infer = time.time() - t_infer_start
            infer_times.append(t_infer)
            if len(infer_times) > 30:
                infer_times.pop(0)
            fps_infer = 1.0 / (sum(infer_times) / len(infer_times))

            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                last_boxes = result.boxes.xyxy.cpu().numpy()
                last_scores = result.boxes.conf.cpu().numpy()
                last_class_ids = result.boxes.cls.cpu().numpy()
            else:
                last_boxes = np.array([])
                last_scores = np.array([])
                last_class_ids = np.array([])

            w_label, w_probs = weather_hook.get_weather_info()
            if w_label is not None:
                last_weather_label = w_label
                last_weather_probs = w_probs
        else:
            reuse_count += 1

        # 可视化
        if len(last_boxes) > 0:
            frame = draw_detections(frame, last_boxes, last_scores,
                                    last_class_ids, class_names)

        elapsed = time.time() - time_start
        fps_overall = frame_idx / elapsed if elapsed > 0 else 0

        frame = draw_hud(
            frame, fps_infer, fps_overall,
            last_weather_label, last_weather_probs,
            frame_idx, is_detection_frame, total_frames
        )

        if writer is not None:
            writer.write(frame)

        if args.show:
            cv2.imshow("Weather-Adaptive Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] 用户中断")
                break
            elif key == ord('s'):
                ss_path = f"screenshot_{frame_idx}.jpg"
                cv2.imwrite(ss_path, frame)
                print(f"[INFO] 截图已保存: {ss_path}")

    # 清理
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    weather_hook.remove()

    # 统计报告
    total_time = time.time() - time_start
    print("\n" + "=" * 60)
    print("推理统计报告")
    print("=" * 60)
    print(f"  总帧数:       {frame_idx}")
    print(f"  检测帧:       {detect_count} ({detect_count / frame_idx * 100:.1f}%)")
    print(f"  复用帧:       {reuse_count} ({reuse_count / frame_idx * 100:.1f}%)")
    print(f"  总耗时:       {total_time:.1f}s")
    print(f"  整体 FPS:     {frame_idx / total_time:.1f}")
    if infer_times:
        avg_infer = sum(infer_times) / len(infer_times)
        print(f"  平均推理耗时: {avg_infer * 1000:.1f}ms/帧")
        print(f"  推理 FPS:     {1 / avg_infer:.1f}")
    print(f"  抽帧间隔:     Δn={delta_n} (原始 {f0:.0f}fps → 检测 {fs}fps)")
    print(f"  计算量节省:   {(1 - 1 / delta_n) * 100:.0f}%")
    print("=" * 60)


# ============================================================
#  入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="天气感知动态跳连网络 — 视频推理管线"
    )
    parser.add_argument("--source", type=str, required=True,
                        help="视频文件路径或摄像头编号 (0, 1, ...)")
    parser.add_argument("--weights", type=str, default="best.pt",
                        help="模型权重路径 (.pt 或 .engine)")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_INPUT_SIZE,
                        help=f"模型输入尺寸 (默认 {DEFAULT_INPUT_SIZE})")
    parser.add_argument("--target-fps", type=int, default=DEFAULT_TARGET_FPS,
                        help=f"抽帧频率 f_s (默认 {DEFAULT_TARGET_FPS})")
    parser.add_argument("--output", type=str, default=None,
                        help="输出视频路径 (不指定则不保存)")
    parser.add_argument("--show", action="store_true", default=True,
                        help="实时显示推理画面 (默认开启)")
    parser.add_argument("--no-show", action="store_true",
                        help="关闭实时显示（服务器/无显示器环境）")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD,
                        help=f"检测置信度阈值 (默认 {DEFAULT_CONF_THRESHOLD})")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD,
                        help=f"NMS IoU 阈值 (默认 {DEFAULT_IOU_THRESHOLD})")

    args = parser.parse_args()
    if args.no_show:
        args.show = False
    return args


if __name__ == "__main__":
    args = parse_args()
    run_video_inference(args)