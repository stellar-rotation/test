"""
两阶段串联推理：壁画 → 线稿提取 → 线稿修复
"""

import sys, cv2, numpy as np, os
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from stage1_extraction.infer import extract_lineart as s1_extract
from stage2_inpaint.infer import restore as s2_restore


def mural_to_lineart(image_path, mask_path=None, ckpt1=None, ckpt2=None, device=None):
    """
    端到端：破损壁画 → 完整线稿
    参数:
        image_path: 壁画原图
        mask_path:  剥落/破损 Mask（可选）
        ckpt1:      阶段一权重
        ckpt2:      阶段二权重
        device:     cpu / cuda
    """
    if device is None:
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    if ckpt1 is None:
        ckpt1 = str(config.CHECKPOINT_DIR / "stage1_best.pt")
    if ckpt2 is None:
        ckpt2 = str(config.CHECKPOINT_DIR / "edge_generator_best.pt")

    out_dir = config.OUTPUT_DIR
    stem = Path(image_path).stem

    # 阶段一：壁画 → 粗线稿
    print(f"[1/2] 线稿提取...")
    raw = s1_extract(image_path, ckpt1, device)
    raw_path = out_dir / f"{stem}_raw.png"
    cv2.imencode(".png", raw)[1].tofile(str(raw_path))

    # 阶段二：有mask才修复
    if mask_path:
        print(f"[2/2] 线稿修复...")
        restored = s2_restore(str(raw_path), mask_path=mask_path,
                              mural_path=image_path, ckpt=ckpt2, device=device)
        final_path = out_dir / f"{stem}_final.png"
        cv2.imencode(".png", restored)[1].tofile(str(final_path))
        print(f"  Raw:      {raw_path}")
        print(f"  Final:    {final_path}")
        return restored
    else:
        print(f"[2/2] 跳过修复（无mask）")
        print(f"  Result:   {raw_path}")
        return raw


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("image"); p.add_argument("-m","--mask",default=None,help="损伤Mask (阶段二修复必需)")
    p.add_argument("--ckpt1",default=None); p.add_argument("--ckpt2",default=None)
    p.add_argument("--cpu",action="store_true")
    a = p.parse_args()
    mural_to_lineart(a.image, a.mask, a.ckpt1, a.ckpt2, device="cpu" if a.cpu else None)
