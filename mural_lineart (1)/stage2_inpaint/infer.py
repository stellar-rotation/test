"""
线稿修复推理：破损线稿 + Mask → 完整线稿
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import cv2, numpy as np, torch
import config
from models.edge_generator import EdgeGenerator


def restore(lineart_path, mask_path=None, mural_path=None, ckpt=None, device=None):
    """修复线稿断裂"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if ckpt is None:
        ckpt = str(config.CHECKPOINT_DIR / "edge_generator_best.pt")

    G = EdgeGenerator(in_ch=config.S2_IN_CH, out_ch=config.S2_OUT_CH).to(device)
    c = torch.load(ckpt, map_location=device, weights_only=False)
    G.load_state_dict(c["model"]); G.eval()

    # 读取线稿
    buf = np.fromfile(lineart_path, dtype=np.uint8)
    lineart = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    h, w = lineart.shape

    # 缩放
    s = config.IMAGE_SIZE / min(h, w)
    if abs(s - 1) > 0.01:
        nh, nw = int(h * s), int(w * s)
        lineart = cv2.resize(lineart, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

    # Mask
    if mask_path:
        buf2 = np.fromfile(mask_path, dtype=np.uint8)
        mask = cv2.imdecode(buf2, cv2.IMREAD_GRAYSCALE)
        if abs(s - 1) > 0.01:
            mask = cv2.resize(mask, (lineart.shape[1], lineart.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        mask = np.zeros_like(lineart)

    # 原图灰度（施加相同mask损伤，防信息泄露）
    if mural_path:
        buf3 = np.fromfile(mural_path, dtype=np.uint8)
        mural = cv2.imdecode(buf3, cv2.IMREAD_COLOR)
        mural_gray = cv2.cvtColor(mural, cv2.COLOR_BGR2GRAY)
        if abs(s - 1) > 0.01:
            mural_gray = cv2.resize(mural_gray, (lineart.shape[1], lineart.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        # Mask区域填灰，消除泄露
        mural_gray[mask > 0] = 230
    else:
        mural_gray = np.full_like(lineart, 128)

    # 归一化拼输入
    lineart_n = lineart.astype(np.float32) / 255.0
    mask_n = mask.astype(np.float32) / 255.0
    mural_n = mural_gray.astype(np.float32) / 255.0
    inp = torch.from_numpy(np.stack([lineart_n, mask_n, mural_n], axis=0)).unsqueeze(0).to(device)

    with torch.no_grad():
        out = G(inp).squeeze().cpu().numpy()
    out = (out * 255).clip(0, 255).astype(np.uint8)

    if abs(s - 1) > 0.01:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return out


if __name__ == "__main__":
    import argparse, os
    p = argparse.ArgumentParser()
    p.add_argument("lineart"); p.add_argument("-m","--mask",default=None)
    p.add_argument("-r","--mural",default=None)
    p.add_argument("-c","--checkpoint",default=None); p.add_argument("-o","--output",default=None)
    a = p.parse_args()
    r = restore(a.lineart, a.mask, a.mural, a.checkpoint)
    out = a.output or a.lineart.rsplit(".",1)[0]+"_restored.png"
    cv2.imencode(".png", r)[1].tofile(out)
    print(f"Saved: {out}")
