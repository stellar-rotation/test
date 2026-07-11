"""阶段一数据集：真实壁画病害Mask + 合成Mask混合 → 线稿"""

import os, random, cv2, numpy as np, torch
from torch.utils.data import Dataset
from pathlib import Path

# 有效mask类别（排除纯色的Crack/Scratch）
_VALID_MASK_CATEGORIES = ["CausticSodaCrystallization", "ChangeColor", "Fading",
                          "FallenOff", "InsectInfestation", "Moldy",
                          "SmokeSmoke", "WaterStains"]


def _load_real_mask_paths(mask_dir):
    """扫描mask目录，返回所有有效mask文件的路径列表"""
    paths = []
    root = Path(mask_dir)
    if not root.is_dir():
        return paths
    for cat in _VALID_MASK_CATEGORIES:
        cat_dir = root / cat
        if cat_dir.is_dir():
            for f in cat_dir.iterdir():
                if f.suffix.lower() == ".png":
                    paths.append(str(f))
    return paths


def _apply_real_mask(img, mask_path, image_size):
    """加载真实mask → threshold → crop/resize → 应用到图像"""
    h_img, w_img = img.shape[:2]

    mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None, None

    # 阈值化：暗像素(<128) = 破损
    binary_mask = (mask < 128).astype(np.float32)

    # 裁切/缩放至目标尺寸
    mh, mw = mask.shape[:2]
    if mh >= image_size and mw >= image_size:
        top = random.randint(0, mh - image_size)
        left = random.randint(0, mw - image_size)
        binary_mask = binary_mask[top:top + image_size, left:left + image_size]
    else:
        binary_mask = cv2.resize(binary_mask, (image_size, image_size),
                                 interpolation=cv2.INTER_NEAREST)

    if binary_mask.sum() < 10:  # 损坏区域太小，跳过
        return None, None

    binary_mask = (binary_mask > 0.5).astype(np.float32)
    fill_color = random.randint(200, 255)
    degraded = (img.astype(np.float32) * (1 - binary_mask[:, :, None]) +
                fill_color * binary_mask[:, :, None])
    degraded = np.clip(degraded, 0, 255).astype(np.uint8)
    return degraded, binary_mask


def degrade_mural(img, severity=0.5):
    """合成破损，返回破损图 + 二值mask"""
    h, w = img.shape[:2]
    damage_mask = np.zeros((h, w), dtype=np.uint8)

    # 多边形块
    if random.random() < 0.75:
        for _ in range(random.randint(1, 3)):
            nv = random.randint(5, 10)
            cx, cy = random.randint(0, w - 1), random.randint(0, h - 1)
            radius = int(60 + 100 * severity)
            pts = []
            for i in range(nv):
                a = 2 * np.pi * i / nv + random.uniform(-0.3, 0.3)
                r = radius * random.uniform(0.5, 1.2)
                pts.append([int(np.clip(cx + r * np.cos(a), 0, w - 1)),
                            int(np.clip(cy + r * np.sin(a), 0, h - 1))])
            cv2.fillPoly(damage_mask, [np.array(pts, dtype=np.int32).reshape((-1, 1, 2))], 255)

    # 圆斑
    if random.random() < 0.5:
        for _ in range(random.randint(3, 12)):
            cv2.circle(damage_mask, (random.randint(0, w - 1), random.randint(0, h - 1)),
                       random.randint(6, int(35 * severity)), 255, -1)

    # 裂缝
    if random.random() < 0.4:
        for _ in range(random.randint(1, 3)):
            cv2.line(damage_mask, (random.randint(0, w - 1), random.randint(0, h - 1)),
                     (random.randint(0, w - 1), random.randint(0, h - 1)), 255, random.randint(3, 8))

    if damage_mask.sum() == 0:
        return img.copy(), np.zeros((h, w), dtype=np.float32)

    binary_mask = (damage_mask > 0).astype(np.float32)
    fill_color = random.randint(200, 255)
    degraded = (img.astype(np.float32) * (1 - binary_mask[:, :, None]) +
                fill_color * binary_mask[:, :, None])
    return np.clip(degraded, 0, 255).astype(np.uint8), binary_mask


class Stage1Dataset(Dataset):
    def __init__(self, image_dir, edge_dir, image_size=512, augment=True,
                 real_mask_dir=None, real_mask_ratio=0.0):
        self.image_dir = image_dir
        self.edge_dir = edge_dir
        self.image_size = image_size
        self.augment = augment
        self.real_mask_ratio = real_mask_ratio

        img_files = set(os.listdir(image_dir))
        edge_files = set(os.listdir(edge_dir))
        self.pairs = sorted(img_files & edge_files)

        self.real_mask_paths = []
        if real_mask_dir and real_mask_ratio > 0:
            self.real_mask_paths = _load_real_mask_paths(real_mask_dir)
            if self.real_mask_paths:
                print(f"Loaded {len(self.real_mask_paths)} real damage masks "
                      f"(ratio={real_mask_ratio:.0%})")
            else:
                print("No real masks found, falling back to synthetic only")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        name = self.pairs[idx]
        img = cv2.imdecode(np.fromfile(os.path.join(self.image_dir, name), dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        edge = cv2.imdecode(np.fromfile(os.path.join(self.edge_dir, name), dtype=np.uint8),
                            cv2.IMREAD_GRAYSCALE)
        if img is None or edge is None:
            raise RuntimeError(f"Failed to load: {name}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        if h >= self.image_size and w >= self.image_size:
            if self.augment:
                top = random.randint(0, h - self.image_size)
                left = random.randint(0, w - self.image_size)
            else:
                top = (h - self.image_size) // 2
                left = (w - self.image_size) // 2
        else:
            s = self.image_size / min(h, w)
            nh, nw = int(h * s), int(w * s)
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
            edge = cv2.resize(edge, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
            top = (nh - self.image_size) // 2
            left = (nw - self.image_size) // 2
        img = img[top:top + self.image_size, left:left + self.image_size]
        edge = edge[top:top + self.image_size, left:left + self.image_size]

        if self.augment and random.random() < 0.5:
            img = cv2.flip(img, 1)
            edge = cv2.flip(edge, 1)

        # 80%概率使用真实mask, 20%使用合成mask
        use_real = (self.real_mask_paths and random.random() < self.real_mask_ratio)
        if use_real:
            mask_path = random.choice(self.real_mask_paths)
            degraded, binary_mask = _apply_real_mask(img, mask_path, self.image_size)
            if degraded is None:
                use_real = False  # 回退到合成

        if not use_real:
            r = random.random()
            sv = random.uniform(0.2, 0.45) if r < 0.5 else random.uniform(0.45, 0.7)
            degraded, binary_mask = degrade_mural(img, severity=sv)

        binary_mask = (binary_mask > 0.5).astype(np.float32)

        # Target A: 残缺线稿
        broken_edge = edge.astype(np.float32) * (1 - binary_mask) + 255.0 * binary_mask
        broken_edge = np.clip(broken_edge, 0, 255).astype(np.uint8)

        # 判别器条件: 挖空破损区的原图灰度
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gray_damaged = img_gray.copy()
        img_gray_damaged[binary_mask > 0] = 230

        degraded_n = degraded.astype(np.float32) / 255.0
        mask_n = binary_mask.astype(np.float32)
        broken_n = broken_edge.astype(np.float32) / 255.0
        perfect_n = edge.astype(np.float32) / 255.0
        gray_n = img_gray_damaged.astype(np.float32) / 255.0

        return {
            "img": torch.from_numpy(degraded_n).permute(2, 0, 1),
            "mask": torch.from_numpy(mask_n).unsqueeze(0),
            "broken_edge": torch.from_numpy(broken_n).unsqueeze(0),
            "perfect_edge": torch.from_numpy(perfect_n).unsqueeze(0),
            "img_gray": torch.from_numpy(gray_n).unsqueeze(0),
        }
