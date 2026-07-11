"""
三联图可视化生成脚本：[Damaged Input] | [Predicted] | [GT Lineart]
用于直观对比模型线稿提取与修复的性能。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import cv2
import numpy as np
import torch
import argparse

import config
from models.pix2pix import ExtractionGenerator
from data.damaged_mural_dataset import degrade_mural
from stage1_extraction.infer import _sliding_infer


def draw_label(img, text, color=(0, 255, 0), scale=1.0, thickness=2):
    """在图像左上角绘制绿色标签文字"""
    # 拷贝一份以防修改原图
    img_cp = img.copy()
    cv2.putText(
        img_cp,
        text,
        (15, 35),  # 文本左下角坐标 (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )
    return img_cp


def generate_visualization(image_path, mask_path=None, gt_path=None, checkpoint_path=None, output_path=None, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if checkpoint_path is None:
        checkpoint_path = str(config.CHECKPOINT_DIR / "stage1_extractor_best.pt")

    # 1. 加载模型
    G = ExtractionGenerator(
        in_ch=config.TRAIN_IN_CH,
        out_ch=config.TRAIN_OUT_CH,
        base_ch=config.STAGE1_BASE_CH,
        num_downs=config.NUM_DOWNS
    ).to(device)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"未找到模型权重文件: {checkpoint_path}，请先确保模型已训练或放置权重。")
        
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    G.load_state_dict(ckpt["model"])
    G.eval()

    # 2. 读取输入图像
    img_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取壁画图像: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img_rgb.shape[:2]

    # 3. 尺寸规范化（缩放至标准训练尺寸基准）
    scale = config.IMAGE_SIZE / min(h_orig, w_orig)
    if abs(scale - 1.0) > 0.01:
        new_h, new_w = int(h_orig * scale), int(w_orig * scale)
        img_rgb_scaled = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        new_h, new_w = h_orig, w_orig
        img_rgb_scaled = img_rgb.copy()

    # 4. 生成或加载损伤 Mask 与 Damaged Input
    if mask_path:
        # 加载用户提供的真实 Mask
        m_gray = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        m_gray = cv2.resize(m_gray, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        # 根据 Mask 模糊填充生成退化图
        degraded_bgr, damage_mask = degrade_mural(cv2.cvtColor(img_rgb_scaled, cv2.COLOR_RGB2BGR), severity=0.5)
        # 用真实的掩膜覆盖
        damage_mask = m_gray
        # 重新融合
        if damage_mask.sum() > 0:
            blurred = cv2.GaussianBlur(cv2.cvtColor(img_rgb_scaled, cv2.COLOR_RGB2BGR), (21, 21), sigmaX=12)
            degraded_bgr[damage_mask > 0] = blurred[damage_mask > 0]
            soft_edge = cv2.GaussianBlur(damage_mask.astype(np.float32), (9, 9), sigmaX=5) / 255.0
            alpha = soft_edge[:, :, None]
            # 修正逻辑：完好区(alpha~0)保留原始清晰图，损坏区(alpha~1)填充模糊图
            degraded_bgr = (degraded_bgr.astype(np.float32) * (1.0 - alpha) + blurred.astype(np.float32) * alpha)
            degraded_bgr = np.clip(degraded_bgr, 0, 255).astype(np.uint8)
        degraded_rgb = cv2.cvtColor(degraded_bgr, cv2.COLOR_BGR2RGB)
    else:
        # 强制循环生成，确保必定产生显著的破损掩膜（重现修复对比效果）
        damage_mask = np.zeros((new_h, new_w), dtype=np.uint8)
        degraded_bgr = cv2.cvtColor(img_rgb_scaled, cv2.COLOR_RGB2BGR)
        for _ in range(50):
            tmp_degraded, tmp_mask = degrade_mural(cv2.cvtColor(img_rgb_scaled, cv2.COLOR_RGB2BGR), severity=0.7)
            if tmp_mask.sum() > 2000:  # 确保有足够面积的损坏区域（如裂纹或大斑块）
                degraded_bgr = tmp_degraded
                damage_mask = tmp_mask
                break
        degraded_rgb = cv2.cvtColor(degraded_bgr, cv2.COLOR_BGR2RGB)

    # 5. 加载真实线稿 GT
    gt_gray = None
    if gt_path:
        gt_gray = cv2.imdecode(np.fromfile(gt_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        gt_gray = cv2.resize(gt_gray, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        # 尝试根据 images 目录自动推断边缘 edges 目录下的同名文件
        possible_gt_paths = [
            image_path.replace("images", "edges"),
            image_path.replace("train\\images", "train\\edges").replace("val\\images", "val\\edges").replace("test\\images", "test\\edges")
        ]
        for p in possible_gt_paths:
            if os.path.exists(p) and p != image_path:
                gt_gray = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                gt_gray = cv2.resize(gt_gray, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                print(f"自动寻找到匹配的 GT 线稿: {p}")
                break

    # 6. 生成预测线稿 (滑窗推理)
    img_n = degraded_rgb.astype(np.float32) / 127.5 - 1.0
    mask_n = damage_mask.astype(np.float32) / 127.5 - 1.0
    img_4ch = np.concatenate([img_n, mask_n[:, :, None]], axis=-1)

    print("正在进行滑窗网络推理...")
    pred_gray = _sliding_infer(G, img_4ch, device)

    # 7. 水平拼接准备（三张图分辨率统一为 new_h x new_w，且全部转为 3 通道 RGB/BGR）
    # 7.1 左图：退化的 RGB 输入
    left_img = cv2.cvtColor(degraded_rgb, cv2.COLOR_RGB2BGR)
    # 7.2 中图：预测的单通道灰度线稿 -> 转成 3 通道 BGR
    mid_img = cv2.cvtColor(pred_gray, cv2.COLOR_GRAY2BGR)
    # 7.3 右图：真值线稿 -> 转成 3 通道 BGR (如果未找到，则用全白图占位)
    if gt_gray is not None:
        right_img = cv2.cvtColor(gt_gray, cv2.COLOR_GRAY2BGR)
    else:
        right_img = np.ones_like(left_img) * 255
        print("未提供且未寻找到 GT 线稿，右侧区域已用白色占位。")

    # 8. 绘制绿色标签文本
    left_labeled = draw_label(left_img, "Damaged Input")
    mid_labeled = draw_label(mid_img, "Predicted")
    right_labeled = draw_label(right_img, "GT Lineart" if gt_gray is not None else "GT Lineart (Not Found)")

    # 9. 水平拼接 (hconcat)
    vis_result = cv2.hconcat([left_labeled, mid_labeled, right_labeled])

    # 10. 保存结果
    if output_path is None:
        os.makedirs(str(config.OUTPUT_DIR), exist_ok=True)
        img_stem = Path(image_path).stem
        output_path = str(config.OUTPUT_DIR / f"{img_stem}_vis.png")

    cv2.imencode(".png", vis_result)[1].tofile(output_path)
    print(f"可视化三联图已成功生成并保存至: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 Damaged | Predicted | GT 三联图的可视化脚本")
    parser.add_argument("image", help="输入原始壁画图片路径")
    parser.add_argument("-m", "--mask", default=None, help="损伤 Mask 路径（可选，不提供则自动合成裂纹）")
    parser.add_argument("-g", "--gt", default=None, help="真实线稿 GT 路径（可选，不提供则自动推导）")
    parser.add_argument("-c", "--checkpoint", default=None, help="指定 checkpoint 路径（可选）")
    parser.add_argument("-o", "--output", default=None, help="指定输出三联图路径（可选）")
    args = parser.parse_args()

    generate_visualization(
        image_path=args.image,
        mask_path=args.mask,
        gt_path=args.gt,
        checkpoint_path=args.checkpoint,
        output_path=args.output
    )
