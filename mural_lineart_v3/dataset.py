"""
阶段一数据集：合成破损壁画 → 线稿 (简单有效)
"""

import os, random, cv2, numpy as np, torch
from torch.utils.data import Dataset


def degrade_mural(img, severity=0.5):
    """合成破损，返回破损图 + 二值mask（修复泄露）"""
    h, w = img.shape[:2]
    damage_mask = np.zeros((h, w), dtype=np.uint8)

    # 多边形块
    if random.random() < 0.75:
        for _ in range(random.randint(1, 3)):
            nv = random.randint(5, 10)
            cx, cy = random.randint(0, w-1), random.randint(0, h-1)
            radius = int(60 + 100 * severity)
            pts = []
            for i in range(nv):
                a = 2 * np.pi * i / nv + random.uniform(-0.3, 0.3)
                r = radius * random.uniform(0.5, 1.2)
                pts.append([int(np.clip(cx+r*np.cos(a),0,w-1)), int(np.clip(cy+r*np.sin(a),0,h-1))])
            cv2.fillPoly(damage_mask, [np.array(pts, dtype=np.int32).reshape((-1,1,2))], 255)

    # 圆斑
    if random.random() < 0.5:
        for _ in range(random.randint(3, 12)):
            cv2.circle(damage_mask, (random.randint(0,w-1), random.randint(0,h-1)),
                       random.randint(6, int(35*severity)), 255, -1)

    # 裂缝
    if random.random() < 0.4:
        for _ in range(random.randint(1, 3)):
            cv2.line(damage_mask, (random.randint(0,w-1), random.randint(0,h-1)),
                     (random.randint(0,w-1), random.randint(0,h-1)), 255, random.randint(3, 8))

    if damage_mask.sum() == 0:
        return img.copy(), np.zeros((h, w), dtype=np.float32)

    # 二值mask硬遮挡（无高斯模糊，无幽灵像素泄露）
    binary_mask = (damage_mask > 0).astype(np.float32)
    fill_color = random.randint(200, 255)
    degraded = (img.astype(np.float32) * (1 - binary_mask[:, :, None]) +
                fill_color * binary_mask[:, :, None])
    return np.clip(degraded, 0, 255).astype(np.uint8), binary_mask


class Stage1Dataset(Dataset):
    def __init__(self, image_dir, edge_dir, image_size=512, augment=True):
        self.image_dir = image_dir
        self.edge_dir = edge_dir
        self.image_size = image_size
        self.augment = augment
        img_files = set(os.listdir(image_dir))
        edge_files = set(os.listdir(edge_dir))
        self.pairs = sorted(img_files & edge_files)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        name = self.pairs[idx]
        img = cv2.imdecode(np.fromfile(os.path.join(self.image_dir, name), dtype=np.uint8), cv2.IMREAD_COLOR)
        edge = cv2.imdecode(np.fromfile(os.path.join(self.edge_dir, name), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        if h >= self.image_size and w >= self.image_size:
            if self.augment:
                top = random.randint(0, h - self.image_size)
                left = random.randint(0, w - self.image_size)
            else:
                top = (h - self.image_size) // 2; left = (w - self.image_size) // 2
        else:
            s = self.image_size / min(h, w)
            nh, nw = int(h * s), int(w * s)
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
            edge = cv2.resize(edge, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
            top = (nh - self.image_size) // 2; left = (nw - self.image_size) // 2
        img = img[top:top+self.image_size, left:left+self.image_size]
        edge = edge[top:top+self.image_size, left:left+self.image_size]

        if self.augment and random.random() < 0.5:
            img = cv2.flip(img, 1); edge = cv2.flip(edge, 1)

        r = random.random()
        sv = random.uniform(0.2, 0.45) if r < 0.5 else random.uniform(0.45, 0.7)
        degraded, binary_mask = degrade_mural(img, severity=sv)

        # Target A: 残缺线稿（和二值mask对齐擦除）
        broken_edge = edge.astype(np.float32) * (1 - binary_mask) + 255.0 * binary_mask
        broken_edge = np.clip(broken_edge, 0, 255).astype(np.uint8)

        # 判别器条件: 挖空破损区的原图灰度
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray_damaged = img_gray.copy()
        img_gray_damaged[binary_mask > 0] = 230

        # 统一归一化到 [0, 1]
        degraded_n = degraded.astype(np.float32) / 255.0
        mask_n = binary_mask.astype(np.float32)  # already 0 or 1
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
