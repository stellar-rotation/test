"""
端到端训练数据集：合成破损壁画，输出 (破损RGB, 损伤Mask, GT线稿)
v4: 支持模糊、白化/脱落和底色遮盖，并软化损伤边缘。
"""

import os, random, cv2, numpy as np, torch
from torch.utils.data import Dataset


_DAMAGE_TYPES = ("blur", "whitening", "substrate")
_DAMAGE_MODES = _DAMAGE_TYPES + ("mixed",)
_DAMAGE_MODE_WEIGHTS = (0.15, 0.25, 0.20, 0.40)
_NEW_DAMAGE_MAX_RATIO = 0.05


def _texture_field(height, width, coarse_scale=32):
    """Generate low-frequency mottling plus fine grain in [-1, 1]."""
    small_h = max(2, height // coarse_scale)
    small_w = max(2, width // coarse_scale)
    coarse = np.random.uniform(-1.0, 1.0, (small_h, small_w)).astype(np.float32)
    coarse = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    coarse = cv2.GaussianBlur(coarse, (0, 0), sigmaX=3.0)
    fine = np.random.normal(0.0, 0.22, (height, width)).astype(np.float32)
    return np.clip(coarse + fine, -1.0, 1.0)


def _sample_damage_mode():
    """Sample blur/whitening/substrate/mixed using deployment-oriented ratios."""
    return random.choices(
        _DAMAGE_MODES,
        weights=_DAMAGE_MODE_WEIGHTS,
        k=1,
    )[0]


def _apply_damage_appearance(image, mask, severity, damage_type):
    """Apply one visual damage type while keeping the supplied binary mask."""
    height, width = mask.shape
    source = image.astype(np.float32)
    severity = float(np.clip(severity, 0.0, 1.0))

    if damage_type == "blur":
        sigma = 5.0 + 8.0 * severity
        target = cv2.GaussianBlur(source, (0, 0), sigmaX=sigma, sigmaY=sigma)
    elif damage_type == "whitening":
        # Chalky pigment loss: warm off-white with uneven fading and fine grain.
        base = np.array(
            [random.uniform(222, 248), random.uniform(218, 244), random.uniform(205, 235)],
            dtype=np.float32,
        )
        texture = _texture_field(height, width, coarse_scale=28)
        chalk = base[None, None, :] + texture[:, :, None] * (10.0 + 8.0 * severity)
        strength = 0.72 + 0.23 * severity
        target = source * (1.0 - strength) + chalk * strength
    elif damage_type == "substrate":
        # Exposed plaster/earth layer. Multiple palettes avoid learning one fixed color.
        palettes = (
            (184, 151, 111),
            (201, 174, 132),
            (166, 133, 101),
            (211, 190, 154),
            (151, 137, 116),
        )
        base = np.array(random.choice(palettes), dtype=np.float32)
        texture = _texture_field(height, width, coarse_scale=22)
        substrate = base[None, None, :] + texture[:, :, None] * (14.0 + 12.0 * severity)

        # Sparse mineral flecks make the fill less like a flat digital paint layer.
        flecks = np.random.random((height, width)).astype(np.float32)
        flecks = cv2.GaussianBlur((flecks > 0.992).astype(np.float32), (3, 3), 0)
        substrate += flecks[:, :, None] * random.choice((-24.0, 24.0))
        strength = 0.82 + 0.16 * severity
        target = source * (1.0 - strength) + substrate * strength
    else:
        raise ValueError(
            f"Unknown damage_type {damage_type!r}; expected one of {_DAMAGE_TYPES}"
        )

    # A soft transition prevents the synthetic boundary from becoming a false edge cue.
    soft_mask = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (9, 9), sigmaX=3.0)
    alpha = np.clip(soft_mask, 0.0, 1.0)[:, :, None]
    result = source * (1.0 - alpha) + target * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def _limit_mask_area(mask, max_ratio):
    """Shrink a binary mask from its boundary until it fits the area limit."""
    target_pixels = int(mask.size * max_ratio)
    current_pixels = int(np.count_nonzero(mask))
    if current_pixels <= target_pixels:
        return mask

    distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    positive_distance = distance[mask > 0]
    threshold_index = max(0, positive_distance.size - target_pixels)
    threshold = np.partition(positive_distance, threshold_index)[threshold_index]

    limited = distance > threshold
    remaining = target_pixels - int(np.count_nonzero(limited))
    if remaining > 0:
        boundary_y, boundary_x = np.where((distance == threshold) & (mask > 0))
        selected = np.random.permutation(boundary_y.size)[:remaining]
        limited[boundary_y[selected], boundary_x[selected]] = True

    return (limited.astype(np.uint8) * 255)


def _generate_scattered_flake_masks(height, width, severity):
    """Generate roughly ten independent irregular flake masks."""
    severity = float(np.clip(severity, 0.0, 1.0))
    masks = []
    flake_count = random.randint(8, 12)
    min_radius = max(6, int(round(6 + 5 * severity)))
    max_radius = max(min_radius + 1, int(round(14 + 12 * severity)))

    for _ in range(flake_count):
        mask = np.zeros((height, width), dtype=np.uint8)
        center_x = random.randint(0, width - 1)
        center_y = random.randint(0, height - 1)
        radius = random.randint(min_radius, max_radius)
        vertex_count = random.randint(6, 10)
        points = []
        for vertex in range(vertex_count):
            angle = 2 * np.pi * vertex / vertex_count + random.uniform(-0.25, 0.25)
            local_radius = radius * random.uniform(0.55, 1.15)
            x = int(np.clip(center_x + local_radius * np.cos(angle), 0, width - 1))
            y = int(np.clip(center_y + local_radius * np.sin(angle), 0, height - 1))
            points.append([x, y])
        polygon = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [polygon], 255)
        masks.append(mask)

    return masks


def _apply_scattered_damage(image, severity, damage_mode):
    """Apply one or several appearances to independent small damage regions."""
    height, width = image.shape[:2]
    flake_masks = _generate_scattered_flake_masks(height, width, severity)

    combined_mask = np.zeros((height, width), dtype=np.uint8)
    for mask in flake_masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    combined_mask = _limit_mask_area(combined_mask, _NEW_DAMAGE_MAX_RATIO)

    if damage_mode == "mixed":
        type_count = 2 if random.random() < 0.75 else 3
        active_types = random.sample(_DAMAGE_TYPES, type_count)
        assignments = active_types.copy()
        assignments.extend(
            random.choice(active_types)
            for _ in range(len(flake_masks) - len(active_types))
        )
        random.shuffle(assignments)
    else:
        assignments = [damage_mode] * len(flake_masks)

    degraded = image.copy()
    for appearance in _DAMAGE_TYPES:
        appearance_mask = np.zeros((height, width), dtype=np.uint8)
        for mask, assigned_type in zip(flake_masks, assignments):
            if assigned_type == appearance:
                appearance_mask = cv2.bitwise_or(appearance_mask, mask)
        appearance_mask = cv2.bitwise_and(appearance_mask, combined_mask)
        if appearance_mask.any():
            degraded = _apply_damage_appearance(
                degraded, appearance_mask, severity, appearance
            )

    return degraded, combined_mask


def degrade_mural(img, severity=0.5, damage_type=None):
    """Synthesize damage; damage_type can also force the mixed mode."""
    if damage_type is not None and damage_type not in _DAMAGE_MODES:
        raise ValueError(
            f"Unknown damage_type {damage_type!r}; expected one of {_DAMAGE_MODES}"
        )

    if damage_type is None:
        damage_type = _sample_damage_mode()

    h, w = img.shape[:2]
    degraded = img.copy()

    # Strong and mixed occlusions use roughly ten small flakes.
    if damage_type in ("whitening", "substrate", "mixed"):
        return _apply_scattered_damage(degraded, severity, damage_type)

    masks = []

    # 1. 大块不规则遮挡
    if random.random() < 0.7:
        n_blocks = random.randint(1, 3)
        for _ in range(n_blocks):
            num_verts = random.randint(5, 10)
            cx, cy = random.randint(0, w-1), random.randint(0, h-1)
            radius = random.randint(int(25 * severity), int(180 * severity))
            pts = []
            for i in range(num_verts):
                angle = 2 * np.pi * i / num_verts + random.uniform(-0.3, 0.3)
                r = radius * random.uniform(0.5, 1.2)
                px = int(np.clip(cx + r * np.cos(angle), 0, w-1))
                py = int(np.clip(cy + r * np.sin(angle), 0, h-1))
                pts.append([px, py])
            pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            masks.append(mask)

    # 2. 中小斑块
    if random.random() < 0.5:
        n_spots = random.randint(3, 12)
        mask = np.zeros((h, w), dtype=np.uint8)
        for _ in range(n_spots):
            cx, cy = random.randint(0, w-1), random.randint(0, h-1)
            r = random.randint(6, int(35 * severity))
            cv2.circle(mask, (cx, cy), r, 255, -1)
        masks.append(mask)

    # 3. 条带状裂缝
    if random.random() < 0.4:
        mask = np.zeros((h, w), dtype=np.uint8)
        for _ in range(random.randint(1, 3)):
            x1, y1 = random.randint(0, w-1), random.randint(0, h-1)
            x2, y2 = random.randint(0, w-1), random.randint(0, h-1)
            thick = random.randint(3, 8)
            cv2.line(mask, (x1, y1), (x2, y2), 255, thick)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=2)
        masks.append(mask)

    # 4. 大面积椭圆/矩形强模糊块（模拟真实壁画大面积脱落/污损）
    if random.random() < 0.35:
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = random.randint(w // 4, 3 * w // 4), random.randint(h // 4, 3 * h // 4)
        if random.random() < 0.5:
            # 大椭圆
            rx = random.randint(int(60 * severity), int(200 * severity))
            ry = random.randint(int(60 * severity), int(200 * severity))
            cv2.ellipse(mask, (cx, cy), (rx, ry), random.uniform(0, 180), 0, 360, 255, -1)
        else:
            # 大矩形
            rw = random.randint(int(80 * severity), int(250 * severity))
            rh = random.randint(int(80 * severity), int(250 * severity))
            x1 = max(0, cx - rw // 2); y1 = max(0, cy - rh // 2)
            x2 = min(w, cx + rw // 2); y2 = min(h, cy + rh // 2)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        masks.append(mask)

    # 合并 mask，并保留原有损伤的通用面积保护。
    damage_mask = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        damage_mask = cv2.bitwise_or(damage_mask, m)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    damage_mask = cv2.dilate(damage_mask, kernel, iterations=1)

    total_pct = (damage_mask > 0).sum() / (h * w)
    if total_pct > 0.65:
        factor = 0.65 / total_pct * random.uniform(0.7, 1.0)
        new_mask = np.zeros((h, w), dtype=np.uint8)
        contours, _ = cv2.findContours(damage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if random.random() < factor:
                cv2.drawContours(new_mask, [cnt], -1, 255, -1)
        damage_mask = new_mask

    # 模糊损伤继续使用原有的大块、斑点和条带掩码。
    if damage_mask.sum() > 0:
        degraded = _apply_damage_appearance(
            degraded, damage_mask, severity, damage_type
        )

    return degraded, damage_mask


class DamagedMuralDataset(Dataset):

    def __init__(self, image_dir, edge_dir, image_size=512, augment=True):
        self.image_dir = image_dir
        self.edge_dir = edge_dir
        self.image_size = image_size
        self.augment = augment
        img_files = set(os.listdir(image_dir))
        edge_files = set(os.listdir(edge_dir))
        self.names = sorted(img_files & edge_files)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img_path = os.path.join(self.image_dir, name)
        edge_path = os.path.join(self.edge_dir, name)

        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        edge = cv2.imdecode(np.fromfile(edge_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
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
            scale = self.image_size / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            edge = cv2.resize(edge, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            top = (new_h - self.image_size) // 2
            left = (new_w - self.image_size) // 2
        img = img[top:top + self.image_size, left:left + self.image_size]
        edge = edge[top:top + self.image_size, left:left + self.image_size]

        if self.augment and random.random() < 0.5:
            img = cv2.flip(img, 1)
            edge = cv2.flip(edge, 1)

        random_state = None
        numpy_random_state = None
        if not self.augment:
            # Keep validation corruption fixed across epochs and workers.
            random_state = random.getstate()
            numpy_random_state = np.random.get_state()
            validation_seed = 1_000_003 + idx
            random.seed(validation_seed)
            np.random.seed(validation_seed % (2**32 - 1))

        try:
            r = random.random()
            if r < 0.3:
                sv = random.uniform(0.2, 0.4)
            elif r < 0.8:
                sv = random.uniform(0.4, 0.7)
            else:
                sv = random.uniform(0.7, 1.0)

            degraded, damage_mask = degrade_mural(img, severity=sv)
        finally:
            if random_state is not None:
                random.setstate(random_state)
                np.random.set_state(numpy_random_state)

        degraded_n = degraded.astype(np.float32) / 127.5 - 1.0
        mask_n = damage_mask.astype(np.float32) / 127.5 - 1.0
        edge_n = edge.astype(np.float32) / 127.5 - 1.0

        inp = np.concatenate([degraded_n.transpose(2, 0, 1), mask_n[None, :, :]], axis=0)
        inp = torch.from_numpy(inp)
        target = torch.from_numpy(edge_n).unsqueeze(0)
        return inp, target
