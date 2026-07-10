"""
阶段一训练完成后，批量推理全部训练图，生成粗线稿供阶段二训练
"""

import sys, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import torch, cv2, numpy as np
from tqdm import tqdm
from pathlib import Path
import config
from models.pix2pix import ExtractionGenerator
from stage1_extraction.infer import extract_lineart


def generate(ckpt=None, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if ckpt is None:
        ckpt = str(config.CHECKPOINT_DIR / "stage1_best.pt")

    out_dir = config.ROOT / "s1_train_outputs"
    out_dir.mkdir(exist_ok=True)

    image_dir = config.TRAIN_IMAGES
    files = sorted(os.listdir(str(image_dir)))

    print(f"Stage 1 inference on {len(files)} training images...")
    for fname in tqdm(files):
        img_path = str(image_dir / fname)
        result = extract_lineart(img_path, ckpt, device)

        out_name = fname.rsplit(".", 1)[0] + ".png"
        cv2.imencode(".png", result)[1].tofile(str(out_dir / out_name))

    print(f"Done: {len(os.listdir(str(out_dir)))} files -> {out_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--checkpoint", default=None)
    args = p.parse_args()
    generate(args.checkpoint)
