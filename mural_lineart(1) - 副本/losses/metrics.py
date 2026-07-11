"""
学术级线稿评估指标：Hausdorff Distance, Chamfer Distance, Edge F1
"""

import numpy as np
from scipy.spatial import cKDTree


def hausdorff_distance(pred, gt, percentile=95):
    """
    豪斯多夫距离（双向，取第95百分位以降噪）
    pred/gt: uint8 二值或灰度线稿 [H, W]
    """
    pred_pts = np.argwhere(_to_binary(pred) > 0)
    gt_pts = np.argwhere(_to_binary(gt) > 0)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("inf")

    tree_pred = cKDTree(pred_pts)
    tree_gt = cKDTree(gt_pts)

    d_gt_to_pred, _ = tree_pred.query(gt_pts)
    d_pred_to_gt, _ = tree_gt.query(pred_pts)

    hd = max(
        np.percentile(d_gt_to_pred, percentile),
        np.percentile(d_pred_to_gt, percentile),
    )
    return hd


def chamfer_distance(pred, gt):
    """倒角距离：双向平均最近邻距离"""
    pred_pts = np.argwhere(_to_binary(pred) > 0)
    gt_pts = np.argwhere(_to_binary(gt) > 0)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("inf")

    tree_pred = cKDTree(pred_pts)
    tree_gt = cKDTree(gt_pts)

    d_gt, _ = tree_pred.query(gt_pts)
    d_pred, _ = tree_gt.query(pred_pts)

    return float(np.mean(d_gt) + np.mean(d_pred))


def edge_f1(pred, gt, tolerance=2):
    """
    边缘 F1-score（带容差）
    tolerance: 像素容差，预测点距 GT 在此距离内算命中
    """
    pred_bin = _to_binary(pred)
    gt_bin = _to_binary(gt)

    pred_pts = np.argwhere(pred_bin > 0)
    gt_pts = np.argwhere(gt_bin > 0)

    if len(pred_pts) == 0 and len(gt_pts) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_pts) == 0:
        return 0.0, 0.0, 0.0
    if len(gt_pts) == 0:
        return 0.0, 0.0, 0.0

    tree_gt = cKDTree(gt_pts)
    d_pred_to_gt, _ = tree_gt.query(pred_pts)

    tree_pred = cKDTree(pred_pts)
    d_gt_to_pred, _ = tree_pred.query(gt_pts)

    tp = (d_pred_to_gt <= tolerance).sum()
    fp = len(pred_pts) - tp
    fn = (d_gt_to_pred > tolerance).sum()

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return float(precision), float(recall), float(f1)


def _to_binary(img, threshold=128):
    if img.dtype == np.bool_:
        return img.astype(np.uint8) * 255
    if img.max() <= 1.0:
        return (img > 0.5).astype(np.uint8) * 255
    return (img > threshold).astype(np.uint8) * 255
