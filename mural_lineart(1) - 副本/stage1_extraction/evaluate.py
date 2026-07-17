"""Evaluate a trained extractor on the paired test split.

The test dataset uses the same deterministic synthetic-damage protocol as
validation. Results are macro-averaged over images and written both per-image
and as a one-row summary CSV.
"""

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from data.damaged_mural_dataset import DamagedMuralDataset
from losses.metrics import evaluate_lineart
from stage1_extraction.infer import _load_generator


METRIC_NAMES = (
    "hole_precision",
    "hole_recall",
    "hole_f1",
    "hole_cldice",
    "hole_hd95",
    "valid_hallucination_rate",
)


def _seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_test_set(checkpoint_path, output_dir=None, device=None, batch_size=4):
    seed = config.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_generator(checkpoint_path, device)
    dataset = DamagedMuralDataset(
        str(config.TEST_IMAGES),
        str(config.TEST_EDGES),
        image_size=config.IMAGE_SIZE,
        augment=False,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=device.type == "cuda",
        worker_init_fn=_seed_worker,
        generator=generator,
    )

    rows = []
    offset = 0
    model.eval()
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            predictions = model(inputs)[:, 0].float().cpu().numpy()
            targets_np = targets[:, 0].float().numpy()
            damage_np = (inputs[:, 3].detach().cpu().numpy() > 0)
            for prediction, target, damage in zip(
                predictions, targets_np, damage_np
            ):
                row = {
                    "image": dataset.names[offset],
                    **evaluate_lineart(prediction, target, damage),
                }
                rows.append(row)
                offset += 1

    if not rows:
        raise RuntimeError("Test dataset is empty")
    summary = {
        "checkpoint": str(Path(checkpoint_path)),
        "seed": seed,
        "image_count": len(rows),
        **{
            name: float(np.mean([row[name] for row in rows]))
            for name in METRIC_NAMES
        },
    }

    output_dir = Path(output_dir) if output_dir else Path(checkpoint_path).parent / "test_results"
    stem = Path(checkpoint_path).stem
    detail_path = output_dir / f"{stem}_seed{seed}_test_metrics.csv"
    summary_path = output_dir / f"{stem}_seed{seed}_test_summary.csv"
    _write_csv(detail_path, rows, ("image", *METRIC_NAMES))
    _write_csv(summary_path, [summary], tuple(summary))
    print(f"Evaluated {len(rows)} test images on {device}")
    print(" | ".join(f"{name}={summary[name]:.6f}" for name in METRIC_NAMES))
    print(f"Per-image metrics: {detail_path}")
    print(f"Summary: {summary_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate line-art restoration")
    parser.add_argument(
        "-c",
        "--checkpoint",
        default=None,
    )
    parser.add_argument(
        "--experiment",
        choices=tuple(config.EXPERIMENT_PRESETS),
        default=config.FINAL_EXPERIMENT,
        help="Used to locate checkpoints/ablations/seed30_<name>/best.pt",
    )
    parser.add_argument("-o", "--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint or (
        config.CHECKPOINT_DIR
        / "ablations"
        / f"seed{config.SEED}_{args.experiment}"
        / "best.pt"
    )
    evaluate_test_set(
        checkpoint_path=checkpoint_path,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
    )
