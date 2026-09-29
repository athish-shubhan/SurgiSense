"""Detection and multi-label metrics. mAP uses COCO 101-point interpolation."""

from __future__ import annotations

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


def _ap_11(scores_matched: list[bool], npos: int) -> float:
    if npos <= 0:
        return float("nan")
    if not scores_matched:
        return 0.0
    tp = 0
    fp = 0
    recs = []
    precs = []
    for hit in scores_matched:
        if hit:
            tp += 1
        else:
            fp += 1
        recs.append(tp / npos)
        precs.append(tp / (tp + fp))
    ap = 0.0
    for i in range(101):
        level = i / 100
        vals = [p for r, p in zip(recs, precs) if r >= level]
        ap += max(vals) if vals else 0.0
    return ap / 101


def _match(preds: list[tuple], gts: dict[str, list[torch.Tensor]], thr: float) -> list[bool]:
    used = {k: set() for k in gts}
    hits = []
    for image_id, _score, box, _lab in preds:
        gt_boxes = gts.get(image_id, [])
        best_iou = 0.0
        best_j = -1
        for j, gt in enumerate(gt_boxes):
            if j in used[image_id]:
                continue
            iou = float(box_iou_matrix(box[None], gt[None])[0, 0])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        ok = best_iou >= thr and best_j >= 0
        if ok:
            used[image_id].add(best_j)
        hits.append(ok)
    return hits


def detection_metrics(samples: list[dict], num_classes: int) -> dict:
    """samples: image_id, pred boxes/scores/labels, gt boxes/labels. Tensors on CPU."""
    iou_thrs = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    per_class_ap = {thr: {} for thr in iou_thrs}
    for cls in range(num_classes):
        preds = []
        gts = {}
        npos = 0
        for sample in samples:
            gid = sample["image_id"]
            gt_mask = sample["gt_labels"] == cls
            gt_boxes = list(sample["gt_boxes"][gt_mask])
            gts[gid] = gt_boxes
            npos += len(gt_boxes)
            pred_mask = sample["pred_labels"] == cls
            for box, score in zip(sample["pred_boxes"][pred_mask], sample["pred_scores"][pred_mask]):
                preds.append((gid, float(score), box, cls))
        preds.sort(key=lambda p: -p[1])
        if npos == 0:
            continue
        for thr in iou_thrs:
            hits = _match(preds, gts, thr)
            per_class_ap[thr][cls] = _ap_11(hits, npos)

    def mean_ap(thr: float) -> float:
        vals = [v for v in per_class_ap[thr].values() if v == v]
        return sum(vals) / len(vals) if vals else float("nan")

    map50 = mean_ap(0.5)
    map_all = sum(mean_ap(thr) for thr in iou_thrs) / len(iou_thrs)
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
