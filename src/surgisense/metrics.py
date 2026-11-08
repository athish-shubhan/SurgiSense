"""Detection and multi-label metrics. mAP uses COCO 101-point interpolation."""

from __future__ import annotations

import numpy as np
import torch


def box_iou_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if len(a) == 0 or len(b) == 0:
        return a.new_zeros((len(a), len(b)))
    tl = torch.max(a[:, None, :2], b[None, :, :2])
    br = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (br - tl).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0))[:, None]
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    return inter / (area_a + area_b - inter + 1e-7)


IOU_THRESHOLDS = np.round(0.5 + 0.05 * np.arange(10), 2)
RECALL_LEVELS = np.arange(101) / 100


def average_precision(hits: np.ndarray, npos: int) -> float:
    """COCO 101-point AP. hits are match flags for predictions in descending score order."""
    if npos <= 0:
        return float("nan")
    if len(hits) == 0:
        return 0.0
    tp = np.cumsum(hits)
    fp = np.cumsum(~hits)
    recall = tp / npos
    precision = tp / (tp + fp)
    # Best precision at this recall or any higher one.
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    first = np.searchsorted(recall, RECALL_LEVELS, side="left")
    reached = first < len(recall)
    return float(envelope[first[reached]].sum() / len(RECALL_LEVELS))


def _greedy_hits(iou: np.ndarray) -> np.ndarray:
    """iou is preds x gts for one image and class, preds in score order.

    Returns preds x thresholds match flags. Each prediction takes the unused
    ground truth with the highest IoU, as in the COCO evaluator.
    """
    hits = np.zeros((iou.shape[0], len(IOU_THRESHOLDS)), dtype=bool)
    if iou.shape[1] == 0:
        return hits
    for t, thr in enumerate(IOU_THRESHOLDS):
        free = np.ones(iou.shape[1], dtype=bool)
        for i in range(iou.shape[0]):
            cand = np.where(free, iou[i], -1.0)
            j = int(cand.argmax())
            if cand[j] >= thr:
                free[j] = False
                hits[i, t] = True
    return hits


def detection_metrics(samples: list[dict], num_classes: int) -> dict:
    """samples: image_id, pred boxes/scores/labels, gt boxes/labels. Tensors on CPU."""
    per_class_ap = {float(thr): {} for thr in IOU_THRESHOLDS}
    for cls in range(num_classes):
        scores, hits, npos = [], [], 0
        for sample in samples:
            gt = sample["gt_boxes"][sample["gt_labels"] == cls]
            npos += len(gt)
            mask = sample["pred_labels"] == cls
            if not mask.any():
                continue
            s = sample["pred_scores"][mask]
            order = s.argsort(descending=True, stable=True)
            iou = box_iou_matrix(sample["pred_boxes"][mask][order], gt).numpy()
            scores.append(s[order].numpy())
            hits.append(_greedy_hits(iou))
        if npos == 0:
            continue
        if scores:
            order = np.argsort(-np.concatenate(scores), kind="stable")
            hits_all = np.concatenate(hits)[order]
        else:
            hits_all = np.zeros((0, len(IOU_THRESHOLDS)), dtype=bool)
        for t, thr in enumerate(IOU_THRESHOLDS):
            per_class_ap[float(thr)][cls] = average_precision(hits_all[:, t], npos)

    def mean_ap(thr: float) -> float:
        vals = [v for v in per_class_ap[thr].values() if v == v]
        return sum(vals) / len(vals) if vals else float("nan")

    map50 = mean_ap(0.5)
    map_all = sum(mean_ap(thr) for thr in per_class_ap) / len(per_class_ap)
    pr = precision_recall(samples, iou_thr=0.5, score_thr=0.5)
    return {
        "map50": map50,
        "map50_95": map_all,
        "per_class_ap50": {str(k): v for k, v in per_class_ap[0.5].items()},
        "precision_at_0.5": pr["precision"],
        "recall_at_0.5": pr["recall"],
        "f1_at_0.5": pr["f1"],
        "tp": pr["tp"],
        "fp": pr["fp"],
        "fn": pr["fn"],
    }


def precision_recall(samples: list[dict], iou_thr: float, score_thr: float) -> dict:
    tp = fp = fn = 0
    for sample in samples:
        order = sample["pred_scores"].argsort(descending=True)
        boxes = sample["pred_boxes"][order]
        scores = sample["pred_scores"][order]
        labels = sample["pred_labels"][order]
        keep = scores >= score_thr
        boxes, labels = boxes[keep], labels[keep]
        matched = set()
        for box, lab in zip(boxes, labels):
            gt_idx = (sample["gt_labels"] == lab).nonzero(as_tuple=False).flatten()
            best_iou, best_j = 0.0, -1
            for j in gt_idx.tolist():
                if j in matched:
                    continue
                iou = float(box_iou_matrix(box[None], sample["gt_boxes"][j][None])[0, 0])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thr and best_j >= 0:
                tp += 1
                matched.add(best_j)
            else:
                fp += 1
        fn += int(len(sample["gt_labels"]) - len(matched))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def multilabel_report(pred: torch.Tensor, target: torch.Tensor, names: list[str]) -> dict:
    """pred and target are N,C probabilities / binary."""
    pred_b = pred >= 0.5
    target_b = target >= 0.5
    per_class = []
    f1s = []
    for c, name in enumerate(names):
        tp = int((pred_b[:, c] & target_b[:, c]).sum())
        fp = int((pred_b[:, c] & ~target_b[:, c]).sum())
        fn = int((~pred_b[:, c] & target_b[:, c]).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = int(target_b[:, c].sum())
        if support == 0 and tp + fp == 0:
            continue
        f1s.append(f1)
        per_class.append(
            {"name": name, "precision": precision, "recall": recall, "f1": f1, "support": support}
        )
    exact = float((pred_b == target_b).all(dim=1).float().mean()) if len(pred) else 0.0
    return {
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "exact_match": exact,
        "per_class": per_class,
    }


def segment_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean per-class temporal IoU. pred/target are T,C bool."""
    ious = []
    for c in range(pred.shape[1]):
        inter = int((pred[:, c] & target[:, c]).sum())
        union = int((pred[:, c] | target[:, c]).sum())
        if union:
            ious.append(inter / union)
    return sum(ious) / len(ious) if ious else float("nan")


def flicker(flags: torch.Tensor) -> float:
    """Mean rate of frame-to-frame label flips. flags is T,C bool."""
    if len(flags) < 2:
        return 0.0
    return float((flags[1:] != flags[:-1]).float().mean())
