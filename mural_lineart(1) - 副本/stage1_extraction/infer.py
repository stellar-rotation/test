"""Stage-1 inference: damaged mural RGB + damage mask -> complete line art."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch

import config
from models.pix2pix import ExtractionGenerator


PATCH_SIZE = config.IMAGE_SIZE
DEFAULT_STRIDE = 384
DEFAULT_BATCH_SIZE = 4


def _read_image(path, flags):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")

    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")
    return image


def _write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"Failed to encode output image: {path}")
    encoded.tofile(path)


def _load_generator(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    model = ExtractionGenerator(
        in_ch=config.TRAIN_IN_CH,
        out_ch=config.TRAIN_OUT_CH,
        base_ch=config.STAGE1_BASE_CH,
        num_downs=config.NUM_DOWNS,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint has no valid model state: {checkpoint_path}")

    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint is incompatible with current ExtractionGenerator: "
            f"{checkpoint_path}"
        ) from exc

    model.eval()
    return model


def _tile_starts(length, patch_size, stride):
    last = length - patch_size
    if last <= 0:
        return [0]

    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def _blend_window(patch_size, margin):
    window = np.ones(patch_size, dtype=np.float32)
    if margin > 0:
        ramp = np.sin(np.linspace(0.0, np.pi / 2.0, margin, dtype=np.float32))
        window[:margin] = ramp
        window[-margin:] = ramp[::-1]
    return window[:, None] * window[None, :]


def _sliding_infer(
    model,
    img_4ch,
    device,
    batch_size=DEFAULT_BATCH_SIZE,
    stride=DEFAULT_STRIDE,
):
    """Run overlap-tile inference on a normalized [H, W, 4] array."""
    if img_4ch.ndim != 3 or img_4ch.shape[2] != config.TRAIN_IN_CH:
        raise ValueError(f"Expected HxWx{config.TRAIN_IN_CH}, got {img_4ch.shape}")
    if stride <= 0 or stride > PATCH_SIZE:
        raise ValueError(f"stride must be in [1, {PATCH_SIZE}]")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    margin = (PATCH_SIZE - stride) // 2
    blend = _blend_window(PATCH_SIZE, margin)

    # Reflect RGB context at image boundaries, but keep outside-mask area valid.
    # In normalized mask space, -1 means undamaged.
    rgb = cv2.copyMakeBorder(
        img_4ch[:, :, :3],
        margin,
        margin,
        margin,
        margin,
        cv2.BORDER_REFLECT_101,
    )
    mask = cv2.copyMakeBorder(
        img_4ch[:, :, 3],
        margin,
        margin,
        margin,
        margin,
        cv2.BORDER_CONSTANT,
        value=-1.0,
    )
    padded = np.concatenate([rgb, mask[:, :, None]], axis=2)
    hp, wp = padded.shape[:2]

    y_starts = _tile_starts(hp, PATCH_SIZE, stride)
    x_starts = _tile_starts(wp, PATCH_SIZE, stride)
    coordinates = [(y, x) for y in y_starts for x in x_starts]

    canvas = np.zeros((hp, wp), dtype=np.float32)
    weight = np.zeros((hp, wp), dtype=np.float32)

    with torch.inference_mode():
        for start in range(0, len(coordinates), batch_size):
            batch_coordinates = coordinates[start : start + batch_size]
            patches = np.stack(
                [
                    padded[y : y + PATCH_SIZE, x : x + PATCH_SIZE]
                    for y, x in batch_coordinates
                ]
            )
            patch_tensor = (
                torch.from_numpy(patches)
                .permute(0, 3, 1, 2)
                .contiguous()
                .to(device, non_blocking=True)
            )
            predictions = model(patch_tensor)[:, 0].float().cpu().numpy()
            predictions = (predictions + 1.0) * 127.5

            for prediction, (y, x) in zip(predictions, batch_coordinates):
                canvas[y : y + PATCH_SIZE, x : x + PATCH_SIZE] += prediction * blend
                weight[y : y + PATCH_SIZE, x : x + PATCH_SIZE] += blend

    output = canvas / np.maximum(weight, 1e-6)
    if margin > 0:
        output = output[margin:-margin, margin:-margin]
    return np.clip(output, 0, 255).astype(np.uint8)


def _simulate_damage(image_rgb, seed=None):
    from data.damaged_mural_dataset import degrade_mural

    if seed is not None:
        import random

        previous_state = random.getstate()
        previous_numpy_state = np.random.get_state()
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
    else:
        previous_state = None
        previous_numpy_state = None

    try:
        for _ in range(20):
            degraded, mask = degrade_mural(image_rgb.copy(), severity=1.0)
            if mask.any():
                return degraded, mask
    finally:
        if previous_state is not None:
            random.setstate(previous_state)
            np.random.set_state(previous_numpy_state)

    mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    height, width = mask.shape
    cv2.ellipse(
        mask,
        (width // 2, height // 2),
        (width // 4, height // 4),
        0,
        0,
        360,
        255,
        -1,
    )
    blurred = cv2.GaussianBlur(image_rgb, (21, 21), sigmaX=12)
    degraded = image_rgb.copy()
    degraded[mask > 0] = blurred[mask > 0]
    return degraded, mask


def _auto_gt_path(image_path):
    image_path = Path(image_path)
    stem = image_path.stem
    for suffix in (
        "_simulated_degraded",
        "_degraded",
        "_damaged",
        "_input",
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    candidate_dirs = [
        config.TRAIN_EDGES,
        config.VAL_EDGES,
        config.TEST_EDGES,
        image_path.parent,
    ]

    for directory in candidate_dirs:
        for suffix in (".jpg", ".png", ".jpeg", ".bmp"):
            candidate = directory / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def _base_stem(image_path):
    stem = Path(image_path).stem
    for suffix in (
        "_simulated_degraded",
        "_degraded",
        "_damaged",
        "_input",
    ):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _auto_mask_path(image_path):
    image_path = Path(image_path)
    stem = image_path.stem
    base_stem = _base_stem(image_path)
    candidates = [
        image_path.with_name(f"{base_stem}_simulated_mask.png"),
        image_path.with_name(f"{base_stem}_mask.png"),
        image_path.with_name(f"{stem}_mask.png"),
        image_path.with_name(f"{base_stem}_mask.jpg"),
        image_path.with_name(f"{stem}_mask.jpg"),
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _default_comparison_path(image_path):
    return config.OUTPUT_DIR / f"{_base_stem(image_path)}_comparison.png"


def _to_bgr_panel(image, size, interpolation=cv2.INTER_AREA):
    if image.ndim == 2:
        panel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        panel = image.copy()

    height, width = size
    if panel.shape[:2] != (height, width):
        panel = cv2.resize(panel, (width, height), interpolation=interpolation)
    return panel


def _draw_label(panel, label):
    labeled = panel.copy()
    cv2.rectangle(labeled, (0, 0), (220, 34), (255, 255, 255), -1)
    cv2.putText(
        labeled,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return labeled


def _make_comparison(damaged_bgr, restored_gray, gt_path=None):
    target_size = damaged_bgr.shape[:2]
    panels = [
        _draw_label(_to_bgr_panel(damaged_bgr, target_size), "Damaged"),
        _draw_label(
            _to_bgr_panel(restored_gray, target_size, cv2.INTER_NEAREST),
            "Restored",
        ),
    ]

    if gt_path is not None:
        gt = _read_image(gt_path, cv2.IMREAD_GRAYSCALE)
        panels.append(
            _draw_label(
                _to_bgr_panel(gt, target_size, cv2.INTER_NEAREST),
                "Ground Truth",
            )
        )

    return np.hstack(panels)


def extract_lineart(
    image_path,
    checkpoint_path=None,
    mask_path=None,
    device=None,
    simulate=False,
    batch_size=DEFAULT_BATCH_SIZE,
    stride=DEFAULT_STRIDE,
    seed=None,
    invert_mask=False,
    gt_path=None,
    comparison_path=None,
):
    """Extract line art. External masks use white for damaged pixels."""
    if simulate and mask_path is not None:
        raise ValueError("--simulate and --mask cannot be used together")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available")

    checkpoint_path = checkpoint_path or (
        config.CHECKPOINT_DIR / "stage1_extractor_best.pt"
    )
    model = _load_generator(checkpoint_path, device)

    image_path = Path(image_path)
    image_bgr = _read_image(image_path, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_height, original_width = image_rgb.shape[:2]

    # Training upsizes small images to a 512px short side, but crops large
    # images without downscaling. Preserve that same scale convention here.
    scale = max(1.0, PATCH_SIZE / min(original_height, original_width))
    if scale > 1.0:
        work_width = round(original_width * scale)
        work_height = round(original_height * scale)
        image_rgb = cv2.resize(
            image_rgb,
            (work_width, work_height),
            interpolation=cv2.INTER_LANCZOS4,
        )
    else:
        work_height, work_width = original_height, original_width

    if simulate:
        print("Generating synthetic damage...")
        image_rgb, mask = _simulate_damage(image_rgb, seed=seed)
    else:
        resolved_mask_path = Path(mask_path) if mask_path is not None else _auto_mask_path(image_path)
        if resolved_mask_path is None:
            mask = np.zeros((work_height, work_width), dtype=np.uint8)
            print("No mask found; using an all-valid mask.")
        else:
            mask = _read_image(resolved_mask_path, cv2.IMREAD_GRAYSCALE)
            print(f"Using mask: {resolved_mask_path}")

        if mask.shape != (work_height, work_width):
            mask = cv2.resize(
                mask,
                (work_width, work_height),
                interpolation=cv2.INTER_NEAREST,
            )
        if invert_mask:
            mask = 255 - mask
        mask = np.where(mask >= 128, 255, 0).astype(np.uint8)

    damaged_rgb_for_compare = image_rgb.copy()
    image_normalized = image_rgb.astype(np.float32) / 127.5 - 1.0
    mask_normalized = mask.astype(np.float32) / 127.5 - 1.0
    model_input = np.concatenate(
        [image_normalized, mask_normalized[:, :, None]],
        axis=2,
    )

    output = _sliding_infer(
        model,
        model_input,
        device,
        batch_size=batch_size,
        stride=stride,
    )

    if scale > 1.0:
        output = cv2.resize(
            output,
            (original_width, original_height),
            interpolation=cv2.INTER_AREA,
        )

    damaged_bgr = cv2.cvtColor(damaged_rgb_for_compare, cv2.COLOR_RGB2BGR)
    if scale > 1.0:
        damaged_bgr = cv2.resize(
            damaged_bgr,
            (original_width, original_height),
            interpolation=cv2.INTER_AREA,
        )

    if comparison_path is not None:
        resolved_gt_path = Path(gt_path) if gt_path is not None else _auto_gt_path(image_path)
        if resolved_gt_path is not None and not resolved_gt_path.is_file():
            raise FileNotFoundError(f"GT image not found: {resolved_gt_path}")

        comparison = _make_comparison(damaged_bgr, output, resolved_gt_path)
        _write_image(comparison_path, comparison)
        if resolved_gt_path is None:
            print(f"Saved comparison without GT: {comparison_path}")
        else:
            print(f"Saved comparison: {comparison_path}")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract complete line art from a damaged mural"
    )
    parser.add_argument("image", help="input damaged mural path")
    parser.add_argument("-m", "--mask", default=None, help="damage mask, white=damaged")
    parser.add_argument("-c", "--checkpoint", default=None, help="model checkpoint")
    parser.add_argument("-o", "--output", default=None, help="output comparison path")
    parser.add_argument("--gt", default=None, help="ground-truth line art path")
    parser.add_argument("--simulate", action="store_true", help="generate synthetic damage")
    parser.add_argument("--seed", type=int, default=None, help="synthetic damage seed")
    parser.add_argument("--invert-mask", action="store_true", help="invert external mask")
    parser.add_argument("--device", default=None, help="for example cuda, cuda:0, or cpu")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    args = parser.parse_args()

    output_path = args.output or _default_comparison_path(args.image)
    extract_lineart(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        mask_path=args.mask,
        device=args.device,
        simulate=args.simulate,
        batch_size=args.batch_size,
        stride=args.stride,
        seed=args.seed,
        invert_mask=args.invert_mask,
        gt_path=args.gt,
        comparison_path=output_path,
    )
