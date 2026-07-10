"""
推理：破损壁画 → 完整线稿 (3ch RGB, 无Mask)
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import cv2, numpy as np, torch
import config
from models.pix2pix import ExtractionGenerator

PS, MARGIN, STRIDE = 512, 64, 384


def _sliding_infer(model, img_rgb, device):
    h, w = img_rgb.shape[:2]
    ramp = np.sin(np.linspace(0, np.pi/2, MARGIN))
    wy, wx = np.ones(PS), np.ones(PS)
    wy[:MARGIN] = ramp; wy[-MARGIN:] = ramp[::-1]
    wx[:MARGIN] = ramp; wx[-MARGIN:] = ramp[::-1]
    blend = wy[:, None] * wx[None, :]

    padded = cv2.copyMakeBorder(img_rgb, MARGIN, MARGIN, MARGIN, MARGIN, cv2.BORDER_REFLECT_101)
    hp, wp = padded.shape[:2]
    hs = max(1, (hp - PS) // STRIDE + 2)
    ws = max(1, (wp - PS) // STRIDE + 2)
    canvas = np.zeros((hp, wp), dtype=np.float32)
    weight = np.zeros((hp, wp), dtype=np.float32)

    for i in range(hs):
        for j in range(ws):
            y1 = min(i * STRIDE, hp - PS); x1 = min(j * STRIDE, wp - PS)
            y1, x1 = max(0, y1), max(0, x1)
            y2, x2 = y1 + PS, x1 + PS
            patch = padded[y1:y2, x1:x2].astype(np.float32) / 127.5 - 1.0
            pt = torch.from_numpy(patch).permute(2,0,1).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(pt).squeeze().cpu().numpy()
            pred = (pred + 1.0) * 127.5
            canvas[y1:y2, x1:x2] += pred * blend
            weight[y1:y2, x1:x2] += blend

    out = canvas / (weight + 1e-6)
    return np.clip(out[MARGIN:hp-MARGIN, MARGIN:wp-MARGIN], 0, 255).astype(np.uint8)


def extract_lineart(image_path, checkpoint_path=None, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if checkpoint_path is None:
        checkpoint_path = str(config.CHECKPOINT_DIR / "stage1_extractor_best.pt")

    G = ExtractionGenerator(in_ch=config.S1_IN_CH, out_ch=config.S1_OUT_CH,
                            base_ch=config.S1_BASE_CH, num_downs=config.S1_NUM_DOWNS).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    G.load_state_dict(ckpt["model"]); G.eval()

    buf = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 缩放
    h_orig, w_orig = img_rgb.shape[:2]
    scale = 512 / min(h_orig, w_orig)
    if abs(scale - 1) > 0.01:
        nh, nw = int(h_orig * scale), int(w_orig * scale)
        img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    out = _sliding_infer(G, img_rgb, device)
    if abs(scale - 1) > 0.01:
        out = cv2.resize(out, (w_orig, h_orig), interpolation=cv2.INTER_LANCZOS4)
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("image"); p.add_argument("-c","--checkpoint",default=None); p.add_argument("-o","--output",default=None)
    a = p.parse_args()
    r = extract_lineart(a.image, a.checkpoint)
    out = a.output or a.image.rsplit(".",1)[0]+"_lineart.png"
    cv2.imencode(".png", r)[1].tofile(out)
    print(f"Saved: {out}")
