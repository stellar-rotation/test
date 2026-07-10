"""
统一推理：壁画 + Mask → 完整线稿 (Y型双分支，取分支B输出)
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import cv2, numpy as np, torch
import config
from models.unified_net import UnifiedGenerator

PS, MARGIN, STRIDE = 512, 64, 384


def extract(image_path, mask_path=None, ckpt=None, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if ckpt is None:
        ckpt = str(config.CHECKPOINT_DIR / "unified_best.pt")

    G = UnifiedGenerator(in_ch=4, base_ch=config.S1_BASE_CH,
                         num_downs=config.S1_NUM_DOWNS, num_res=config.UNIFIED_NUM_RES).to(device)
    c = torch.load(ckpt, map_location=device, weights_only=True)
    G.load_state_dict(c["model_G"] if "model_G" in c else c["model"]); G.eval()

    # 读取原图 + mask
    buf = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if mask_path:
        buf2 = np.fromfile(mask_path, dtype=np.uint8)
        mask = cv2.imdecode(buf2, cv2.IMREAD_GRAYSCALE)
    else:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

    h_orig, w_orig = img_rgb.shape[:2]
    s = PS / min(h_orig, w_orig)
    if abs(s - 1) > 0.01:
        nh, nw = int(h_orig * s), int(w_orig * s)
        img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    else:
        nh, nw = h_orig, w_orig

    # 归一化 [0,1]
    img_n = img_rgb.astype(np.float32) / 255.0
    mask_n = (mask > 128).astype(np.float32)

    # 滑窗推理
    ramp = np.sin(np.linspace(0, np.pi / 2, MARGIN))
    wy, wx = np.ones(PS), np.ones(PS)
    wy[:MARGIN] = ramp; wy[-MARGIN:] = ramp[::-1]
    wx[:MARGIN] = ramp; wx[-MARGIN:] = ramp[::-1]
    blend = wy[:, None] * wx[None, :]

    padded_img = cv2.copyMakeBorder(img_n, MARGIN, MARGIN, MARGIN, MARGIN, cv2.BORDER_REFLECT_101)
    padded_msk = cv2.copyMakeBorder(mask_n, MARGIN, MARGIN, MARGIN, MARGIN, cv2.BORDER_CONSTANT, value=0)
    hp, wp = padded_img.shape[:2]
    hs = max(1, (hp - PS) // STRIDE + 2)
    ws = max(1, (wp - PS) // STRIDE + 2)
    canvas = np.zeros((hp, wp), dtype=np.float32)
    weight = np.zeros((hp, wp), dtype=np.float32)

    for i in range(hs):
        for j in range(ws):
            y1 = min(i * STRIDE, hp - PS); x1 = min(j * STRIDE, wp - PS)
            y1, x1 = max(0, y1), max(0, x1)
            y2, x2 = y1 + PS, x1 + PS

            patch_img = padded_img[y1:y2, x1:x2]
            patch_msk = padded_msk[y1:y2, x1:x2]

            pt_img = torch.from_numpy(patch_img).permute(2,0,1).unsqueeze(0).to(device)
            pt_msk = torch.from_numpy(patch_msk).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                _, pred = G(pt_img, pt_msk)
            pred = pred.squeeze().cpu().numpy()
            canvas[y1:y2, x1:x2] += pred * blend
            weight[y1:y2, x1:x2] += blend

    out = canvas / (weight + 1e-6)
    out = np.clip(out[MARGIN:hp-MARGIN, MARGIN:wp-MARGIN] * 255, 0, 255).astype(np.uint8)
    if abs(s - 1) > 0.01:
        out = cv2.resize(out, (w_orig, h_orig), interpolation=cv2.INTER_LANCZOS4)
    return out


if __name__ == "__main__":
    import argparse, os
    p = argparse.ArgumentParser()
    p.add_argument("image"); p.add_argument("-m","--mask",default=None)
    p.add_argument("-c","--checkpoint",default=None); p.add_argument("-o","--output",default=None)
    a = p.parse_args()
    r = extract(a.image, a.mask, a.checkpoint)
    out = a.output or a.image.rsplit(".",1)[0]+"_lineart.png"
    cv2.imencode(".png", r)[1].tofile(out)
    print(f"Saved: {out}")
