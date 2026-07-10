"""
阶段二数据集：阶段一粗线稿 + Mask + 原图灰度 → GT线稿
训练时用阶段一真实输出，消除训练-推理域差
"""

import os, random, cv2, numpy as np, torch
from torch.utils.data import Dataset


class Stage2Dataset(Dataset):

    def __init__(self, image_dir, edge_dir, mask_dir, s1_output_dir=None,
                 image_size=512, augment=True):
        self.image_dir = image_dir
        self.edge_dir = edge_dir
        self.mask_dir = mask_dir
        self.s1_output_dir = s1_output_dir
        self.image_size = image_size
        self.augment = augment

        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        img_files = set(os.listdir(image_dir))
        edge_files = set(os.listdir(edge_dir))
        self.pairs = sorted(img_files & edge_files)

        # 真实粗线稿模式：匹配阶段一输出文件名
        self.use_s1 = s1_output_dir and os.path.isdir(s1_output_dir)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        name = self.pairs[idx]
        stem = name.rsplit(".", 1)[0]

        # 原图灰度
        img = cv2.imdecode(np.fromfile(os.path.join(self.image_dir, name), dtype=np.uint8), cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # GT线稿
        edge = cv2.imdecode(np.fromfile(os.path.join(self.edge_dir, name), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        # 阶段一粗线稿
        if self.use_s1:
            s1_path = os.path.join(self.s1_output_dir, stem + ".png")
            if os.path.exists(s1_path):
                rough = cv2.imdecode(np.fromfile(s1_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            else:
                rough = edge.copy()  # fallback
        else:
            rough = edge.copy()

        # 随机 mask
        mask_name = random.choice(self.mask_files)
        mask = cv2.imdecode(np.fromfile(os.path.join(self.mask_dir, mask_name), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        # 尺寸处理
        h, w = rough.shape
        if h >= self.image_size and w >= self.image_size:
            if self.augment:
                top = random.randint(0, h - self.image_size)
                left = random.randint(0, w - self.image_size)
            else:
                top = (h - self.image_size) // 2; left = (w - self.image_size) // 2
        else:
            s = self.image_size / min(h, w)
            nh, nw = int(h * s), int(w * s)
            rough = cv2.resize(rough, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
            edge = cv2.resize(edge, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
            img_gray = cv2.resize(img_gray, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
            top = (nh - self.image_size) // 2; left = (nw - self.image_size) // 2

        rough = rough[top:top+self.image_size, left:left+self.image_size]
        edge = edge[top:top+self.image_size, left:left+self.image_size]
        img_gray = img_gray[top:top+self.image_size, left:left+self.image_size]

        if mask.shape[:2] != (self.image_size, self.image_size):
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        if self.augment and random.random() < 0.5:
            rough = cv2.flip(rough, 1); edge = cv2.flip(edge, 1)
            img_gray = cv2.flip(img_gray, 1); mask = cv2.flip(mask, 1)

        # 施加 mask 到粗线稿和原图灰度（防信息泄露）
        damaged = rough.copy()
        damaged[mask > 0] = 255
        img_damaged = img_gray.copy()
        img_damaged[mask > 0] = 230  # 原图灰度的mask区也填灰

        # 归一化 [0, 1]
        damaged_n = damaged.astype(np.float32) / 255.0
        mask_n = mask.astype(np.float32) / 255.0
        edge_n = edge.astype(np.float32) / 255.0
        img_n = img_damaged.astype(np.float32) / 255.0

        inp = torch.from_numpy(np.stack([damaged_n, mask_n, img_n], axis=0))
        target = torch.from_numpy(edge_n).unsqueeze(0)
        return inp, target
