"""Offline loss-gradient diagnostics for trained line-art checkpoints.

This script never updates model parameters. It evaluates a fixed validation
subset and records raw/weighted gradient norms plus pairwise cosine similarity.
"""

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from data.damaged_mural_dataset import DamagedMuralDataset
from losses.direction_loss import DirectionLoss
from losses.gradient_loss import GradientLoss
from losses.skeleton_loss import SkeletonLoss
from losses.tversky_loss import MaskedTverskyLoss
from models.pix2pix import ExtractionGenerator, PatchGANDiscriminator
from stage1_extraction.train import weighted_per_sample_mean


LOSS_WEIGHTS = {
    "l1": config.LAMBDA_L1,
    "edge": config.LAMBDA_EDGE,
    "ssim": config.LAMBDA_SSIM,
    "tversky": config.LAMBDA_TVERSKY,
    "skeleton": config.LAMBDA_SKEL,
    "direction": config.LAMBDA_DIR,
    "gan": config.LAMBDA_ADV,
}


def _load_models(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    generator = ExtractionGenerator(
        in_ch=config.TRAIN_IN_CH,
        out_ch=config.TRAIN_OUT_CH,
        base_ch=config.STAGE1_BASE_CH,
        num_downs=config.NUM_DOWNS,
    ).to(device)
    generator.load_state_dict(checkpoint["model"])
    generator.eval()

    discriminator = None
    if "D" in checkpoint:
        discriminator = PatchGANDiscriminator(
            in_ch=config.TRAIN_IN_CH + config.TRAIN_OUT_CH
        ).to(device)
        discriminator.load_state_dict(checkpoint["D"])
        discriminator.eval()
        for parameter in discriminator.parameters():
            parameter.requires_grad_(False)
    return checkpoint, generator, discriminator


def _selected_parameters(model, scope):
    if scope == "out_conv":
        parameters = tuple(model.out_conv.parameters())
    elif scope == "bottleneck":
        parameters = tuple(model.bottleneck.parameters())
    elif scope == "all":
        parameters = tuple(model.parameters())
    else:
        raise ValueError(f"Unknown parameter scope: {scope}")
    return tuple(parameter for parameter in parameters if parameter.requires_grad)


def _gradient_vector(loss, parameters, retain_graph):
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    pieces = []
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            pieces.append(torch.zeros_like(parameter).reshape(-1))
        else:
            pieces.append(gradient.detach().float().reshape(-1))
    return torch.cat(pieces)


def _cosine(left, right):
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if denominator.item() == 0.0:
        return float("nan")
    return float(torch.dot(left, right).item() / denominator.item())


def _losses_for_batch(generator, discriminator, inputs, targets, criteria, flags):
    predictions = generator(inputs)
    damage_mask = (inputs[:, 3:4] > 0).float()
    target_line = (targets < 0).float()
    line_focus = torch.nn.functional.max_pool2d(
        target_line,
        kernel_size=config.LINE_FOCUS_KERNEL,
        stride=1,
        padding=config.LINE_FOCUS_KERNEL // 2,
    )
    hole_weights = (
        config.HOLE_BACKGROUND_WEIGHT
        + (config.HOLE_LINE_WEIGHT - config.HOLE_BACKGROUND_WEIGHT) * line_focus
    )
    pixel_weights = (
        config.VALID_RECON_WEIGHT * (1.0 - damage_mask)
        + hole_weights * damage_mask
    )

    losses = {}
    losses["l1"] = weighted_per_sample_mean(
        torch.abs(predictions - targets), pixel_weights
    )

    grad_pred = criteria["edge"]._gradient(predictions)
    grad_target = criteria["edge"]._gradient(targets)
    losses["edge"] = weighted_per_sample_mean(
        torch.abs(grad_pred - grad_target), pixel_weights
    )

    if flags["ssim"]:
        c1, c2 = 0.01**2, 0.03**2
        pred_01 = (predictions + 1.0) / 2.0
        target_01 = (targets + 1.0) / 2.0
        window = 11
        mu_pred = torch.nn.functional.avg_pool2d(
            pred_01, window, 1, window // 2
        )
        mu_target = torch.nn.functional.avg_pool2d(
            target_01, window, 1, window // 2
        )
        pred_sq, target_sq = mu_pred**2, mu_target**2
        pred_target = mu_pred * mu_target
        var_pred = (
            torch.nn.functional.avg_pool2d(pred_01**2, window, 1, window // 2)
            - pred_sq
        )
        var_target = (
            torch.nn.functional.avg_pool2d(target_01**2, window, 1, window // 2)
            - target_sq
        )
        covariance = (
            torch.nn.functional.avg_pool2d(
                pred_01 * target_01, window, 1, window // 2
            )
            - pred_target
        )
        ssim_map = (
            (2 * pred_target + c1) * (2 * covariance + c2)
        ) / (
            (pred_sq + target_sq + c1)
            * (var_pred + var_target + c2)
            + 1e-8
        )
        losses["ssim"] = weighted_per_sample_mean(
            1.0 - ssim_map, pixel_weights
        )

    if flags["tversky"]:
        losses["tversky"] = criteria["tversky"](
            predictions, targets, damage_mask
        )
    if flags["skeleton"]:
        losses["skeleton"] = criteria["skeleton"](predictions, targets)
    if flags["direction"]:
        losses["direction"] = criteria["direction"](predictions, targets)
    if flags["gan"]:
        if discriminator is None:
            raise ValueError("GAN gradient requested, but checkpoint has no D state")
        fake_pair = torch.cat([inputs, predictions], dim=1)
        losses["gan"] = ((discriminator(fake_pair) - 1.0) ** 2).mean()
    return losses


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_checkpoint(
    checkpoint_path,
    output_dir=None,
    device=None,
    batch_size=2,
    max_batches=4,
    scope="out_conv",
    include_skeleton=False,
):
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    checkpoint_path = Path(checkpoint_path)
    checkpoint, generator, discriminator = _load_models(checkpoint_path, device)
    checkpoint_flags = checkpoint.get("loss_flags", {})
    flags = {
        "ssim": checkpoint_flags.get("ssim", True),
        "tversky": checkpoint_flags.get("tversky", True),
        "skeleton": checkpoint_flags.get("skeleton", True) or include_skeleton,
        "direction": checkpoint_flags.get("direction", True),
        "gan": checkpoint_flags.get("gan", True) and discriminator is not None,
    }
    parameters = _selected_parameters(generator, scope)
    parameter_count = sum(parameter.numel() for parameter in parameters)

    criteria = {
        "edge": GradientLoss().to(device),
        "tversky": MaskedTverskyLoss(
            alpha=config.TVERSKY_ALPHA,
            beta=config.TVERSKY_BETA,
            temperature=config.TVERSKY_TEMPERATURE,
        ).to(device),
        "skeleton": SkeletonLoss().to(device),
        "direction": DirectionLoss().to(device),
    }
    dataset = DamagedMuralDataset(
        str(config.VAL_IMAGES),
        str(config.VAL_EDGES),
        image_size=config.IMAGE_SIZE,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    norm_rows = []
    cosine_sums = {}
    cosine_counts = {}
    active_names = None
    for batch_index, (inputs, targets) in enumerate(loader):
        if batch_index >= max_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        losses = _losses_for_batch(
            generator, discriminator, inputs, targets, criteria, flags
        )
        active_names = tuple(losses)
        vectors = {}
        for index, name in enumerate(active_names):
            vectors[name] = _gradient_vector(
                losses[name],
                parameters,
                retain_graph=index < len(active_names) - 1,
            )
            raw_norm = float(torch.linalg.vector_norm(vectors[name]).item())
            norm_rows.append(
                {
                    "batch": batch_index,
                    "loss": name,
                    "raw_loss": float(losses[name].detach().item()),
                    "weight": LOSS_WEIGHTS[name],
                    "raw_grad_norm": raw_norm,
                    "weighted_grad_norm": raw_norm * LOSS_WEIGHTS[name],
                }
            )

        for left in active_names:
            for right in active_names:
                value = _cosine(vectors[left], vectors[right])
                if not math.isnan(value):
                    key = (left, right)
                    cosine_sums[key] = cosine_sums.get(key, 0.0) + value
                    cosine_counts[key] = cosine_counts.get(key, 0) + 1
        print(f"Analyzed batch {batch_index + 1}/{max_batches}")

    if not norm_rows:
        raise RuntimeError("No validation batches were analyzed")
    output_dir = Path(output_dir or checkpoint_path.parent / "gradient_analysis" / checkpoint_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "gradient_norms_by_batch.csv",
        norm_rows,
        (
            "batch", "loss", "raw_loss", "weight",
            "raw_grad_norm", "weighted_grad_norm",
        ),
    )

    summary_rows = []
    for name in active_names:
        rows = [row for row in norm_rows if row["loss"] == name]
        summary_rows.append(
            {
                "loss": name,
                "weight": LOSS_WEIGHTS[name],
                "mean_raw_loss": float(np.mean([row["raw_loss"] for row in rows])),
                "mean_raw_grad_norm": float(
                    np.mean([row["raw_grad_norm"] for row in rows])
                ),
                "mean_weighted_grad_norm": float(
                    np.mean([row["weighted_grad_norm"] for row in rows])
                ),
            }
        )
    _write_csv(
        output_dir / "gradient_norms_summary.csv",
        summary_rows,
        (
            "loss", "weight", "mean_raw_loss",
            "mean_raw_grad_norm", "mean_weighted_grad_norm",
        ),
    )

    cosine_rows = []
    for left in active_names:
        row = {"loss": left}
        for right in active_names:
            key = (left, right)
            row[right] = cosine_sums[key] / cosine_counts[key]
        cosine_rows.append(row)
    _write_csv(
        output_dir / "gradient_cosine_matrix.csv",
        cosine_rows,
        ("loss", *active_names),
    )

    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "experiment": checkpoint.get("experiment"),
        "seed": config.SEED,
        "device": str(device),
        "parameter_scope": scope,
        "parameter_count": parameter_count,
        "batch_size": batch_size,
        "batch_count": max_batches,
        "validation_image_count": batch_size * max_batches,
        "active_losses": list(active_names),
        "loss_flags": flags,
    }
    with (output_dir / "analysis_config.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
    print(f"Gradient analysis saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure loss gradient norms and cosine similarity"
    )
    parser.add_argument("-c", "--checkpoint", default=str(config.FINAL_CHECKPOINT))
    parser.add_argument("-o", "--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument(
        "--scope", choices=("out_conv", "bottleneck", "all"), default="out_conv"
    )
    parser.add_argument(
        "--include-skeleton",
        action="store_true",
        help="Also measure hypothetical Skeleton gradients on a no-Skeleton model",
    )
    args = parser.parse_args()
    analyze_checkpoint(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        scope=args.scope,
        include_skeleton=args.include_skeleton,
    )
