"""Minimal objective metrics for black-line-on-white restoration.

The ablation protocol intentionally keeps four complementary metrics:

* hole_precision/recall/f1: tolerant line detection inside the damage;
* hole_cldice: skeleton-aware topology and continuity near the repair;
* hole_hd95: robust worst-case displacement inside the damaged region;
* valid_hallucination_rate: unsupported skeleton length outside the damage.

Dark pixels are line foreground. Images may use [0, 255], [0, 1], or the
model range [-1, 1]. Region masks use non-zero pixels as True.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def to_line_mask(image, threshold=0.5):
    """Convert a 2-D line-art image to a dark-line boolean mask."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D line-art image, got {image.shape}")
    if image.dtype == np.bool_:
        return image.copy()
    if not np.isfinite(image).all():
        raise ValueError("Line-art image contains NaN or infinity")

    values = image.astype(np.float32, copy=False)
    minimum = float(values.min()) if values.size else 0.0
    maximum = float(values.max()) if values.size else 0.0
    if minimum < 0.0:
        cutoff = threshold * 2.0 - 1.0
    elif maximum <= 1.0:
        cutoff = threshold
    else:
        cutoff = threshold * 255.0
    return values < cutoff


def _region_mask(region, shape):
    if region is None:
        return np.ones(shape, dtype=bool)
    region = np.asarray(region)
    if region.shape != shape:
        raise ValueError(f"Region shape {region.shape} does not match image {shape}")
    return region.astype(bool)


def edge_f1(pred, gt, tolerance=2, region=None, threshold=0.5):
    """Tolerance-aware line precision, recall and F1 inside ``region``."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    pred_line = to_line_mask(pred, threshold)
    gt_line = to_line_mask(gt, threshold)
    region = _region_mask(region, pred_line.shape)
    pred_points = np.argwhere(pred_line & region)
    gt_points = np.argwhere(gt_line & region)

    if len(pred_points) == 0 and len(gt_points) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_points) == 0 or len(gt_points) == 0:
        return 0.0, 0.0, 0.0

    precision = float(
        np.mean(cKDTree(gt_points).query(pred_points)[0] <= tolerance)
    )
    recall = float(
        np.mean(cKDTree(pred_points).query(gt_points)[0] <= tolerance)
    )
    denominator = precision + recall
    f1 = 2.0 * precision * recall / denominator if denominator else 0.0
    return precision, recall, float(f1)


def _skeleton(line_mask):
    image = line_mask.astype(bool, copy=True)
    skeleton = np.zeros_like(image)
    structure = ndimage.generate_binary_structure(2, 1)
    while image.any():
        eroded = ndimage.binary_erosion(image, structure=structure)
        opened = ndimage.binary_dilation(eroded, structure=structure)
        skeleton |= image & ~opened
        image = eroded
    return skeleton


def cldice(pred, gt, region=None, threshold=0.5, smooth=1.0):
    """Hard clDice topology score, optionally restricted after skeletonizing."""
    pred_line = to_line_mask(pred, threshold)
    gt_line = to_line_mask(gt, threshold)
    pred_skeleton = _skeleton(pred_line)
    gt_skeleton = _skeleton(gt_line)
    region = _region_mask(region, pred_line.shape)
    topology_precision = (
        np.count_nonzero(pred_skeleton & gt_line & region) + smooth
    ) / (np.count_nonzero(pred_skeleton & region) + smooth)
    topology_recall = (
        np.count_nonzero(gt_skeleton & pred_line & region) + smooth
    ) / (np.count_nonzero(gt_skeleton & region) + smooth)
    denominator = topology_precision + topology_recall
    return float(
        2.0 * topology_precision * topology_recall / denominator
        if denominator else 0.0
    )


def hausdorff95(pred, gt, region=None, threshold=0.5):
    """Symmetric 95th-percentile Hausdorff distance in pixels."""
    pred_line = to_line_mask(pred, threshold)
    gt_line = to_line_mask(gt, threshold)
    region = _region_mask(region, pred_line.shape)
    pred_points = np.argwhere(pred_line & region)
    gt_points = np.argwhere(gt_line & region)
    if len(pred_points) == 0 and len(gt_points) == 0:
        return 0.0
    if len(pred_points) == 0 or len(gt_points) == 0:
        # A finite image-diagonal penalty keeps dataset means well-defined.
        return float(np.hypot(*pred_line.shape))
    gt_to_pred = cKDTree(pred_points).query(gt_points)[0]
    pred_to_gt = cKDTree(gt_points).query(pred_points)[0]
    return float(max(np.percentile(gt_to_pred, 95), np.percentile(pred_to_gt, 95)))


def valid_hallucination_rate(
    pred, gt, damage_mask, tolerance=2, threshold=0.5
):
    """Fraction of predicted valid-region skeleton unsupported by GT lines.

    A predicted skeleton pixel is hallucinated when it lies in the undamaged
    region and farther than ``tolerance`` pixels from every GT line pixel.
    Using skeleton length avoids dilution by the large white background and
    makes the score insensitive to predicted line thickness.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    pred_line = to_line_mask(pred, threshold)
    gt_line = to_line_mask(gt, threshold)
    damage = _region_mask(damage_mask, pred_line.shape)
    valid_pred_skeleton = _skeleton(pred_line) & ~damage
    predicted_length = int(valid_pred_skeleton.sum())
    if predicted_length == 0:
        return 0.0

    # distance_transform_edt gives each non-GT pixel its distance to GT.
    # If GT is empty, every predicted valid-region line is unsupported.
    if not gt_line.any():
        return 1.0
    distance_to_gt = ndimage.distance_transform_edt(~gt_line)
    hallucinated = valid_pred_skeleton & (distance_to_gt > tolerance)
    return float(hallucinated.sum() / predicted_length)


def evaluate_lineart(pred, gt, damage_mask, tolerance=2, threshold=0.5):
    """Return the four metrics used to compare ablation experiments."""
    damage = _region_mask(damage_mask, np.asarray(pred).shape)
    hole_precision, hole_recall, hole_f1 = edge_f1(
        pred, gt, tolerance=tolerance, region=damage, threshold=threshold
    )
    return {
        "hole_precision": hole_precision,
        "hole_recall": hole_recall,
        "hole_f1": hole_f1,
        "hole_cldice": cldice(pred, gt, damage, threshold),
        "hole_hd95": hausdorff95(pred, gt, damage, threshold),
        "valid_hallucination_rate": valid_hallucination_rate(
            pred, gt, damage, tolerance=tolerance, threshold=threshold
        ),
    }
