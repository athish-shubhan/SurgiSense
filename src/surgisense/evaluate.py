"""Evaluate detectors, temporal models, ablations, and failure modes."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from surgisense.context import assistant_command, build_state
from surgisense.data import CLASSES, FrameDataset, collate, load_config, load_index, records_for_split
from surgisense.detector import ToolDetector, predict
from surgisense.experiment import device, hardware
from surgisense.metrics import detection_metrics, flicker, multilabel_report, segment_iou
from surgisense.temporal import FramePresence, TemporalPresence

OUT = Path("evaluation")


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float):
        return None if obj != obj else obj
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if v != v else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def load_detector(path: Path, dev) -> tuple[torch.nn.Module, dict]:
    if not path.exists():
        raise SystemExit(f"missing checkpoint {path}. Train it before evaluation.")
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    model = ToolDetector(len(CLASSES), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(dev).eval()
    return model, ckpt


def collect(model, records, dev, cfg) -> list[dict]:
    size = (cfg["image_height"], cfg["image_width"])
    loader = DataLoader(
        FrameDataset(records, augment=False, image_size=size),
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=collate,
    )
    samples = []
    for images, batch in loader:
        dets = predict(
            model(images.to(dev)),
            (cfg["image_height"], cfg["image_width"]),
            stride=cfg["stride"],
            score_thresh=cfg["score_thresh"],
            nms_iou=cfg["nms_iou"],
            max_det=cfg["max_det"],
        )
        for det, sample in zip(dets, batch):
            samples.append(
                {
                    "image_id": f"{sample['video']}:{sample['frame']}",
                    "video": sample["video"],
                    "frame": sample["frame"],
                    "file": sample["file"],
                    "pred_boxes": det["boxes"],
                    "pred_scores": det["scores"],
                    "pred_labels": det["labels"],
                    "gt_boxes": sample["boxes"],
                    "gt_labels": sample["labels"],
                }
            )
    return samples


def load_temporal(path: Path, dev):
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    if ckpt["kind"] == "frame":
        model = FramePresence(ckpt["dim"], len(CLASSES))
    else:
        model = TemporalPresence(ckpt["dim"], ckpt["hidden"], len(CLASSES))
    model.load_state_dict(ckpt["model"])
    model.to(dev).eval()
    return model, ckpt


@torch.no_grad()
def presence_probs(model, ckpt, records, feats, dev) -> torch.Tensor:
    probs = torch.zeros(len(records), len(CLASSES))
    if ckpt["kind"] == "frame":
        for i in range(0, len(feats), 64):
            probs[i : i + 64] = torch.sigmoid(model(feats[i : i + 64].to(dev))).cpu()
        return probs
    window = int(ckpt["window"])
    by_video: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        by_video.setdefault(rec["video"], []).append(i)
    for idxs in by_video.values():
        idxs = sorted(idxs, key=lambda i: records[i]["frame"])
        for t, i in enumerate(idxs):
            start = max(0, t - window + 1)
            seq = feats[idxs[start : t + 1]]
            if len(seq) < window:
                seq = torch.cat([seq[:1].repeat(window - len(seq), 1), seq], 0)
            probs[i] = torch.sigmoid(model(seq[None].to(dev)))[0].cpu()
    return probs


def rescale(samples, probs) -> list[dict]:
    gated = []
    for sample, prob in zip(samples, probs):
        if len(sample["pred_scores"]):
            scores = sample["pred_scores"] * prob[sample["pred_labels"]]
        else:
            scores = sample["pred_scores"]
        gated.append({**sample, "pred_scores": scores})
    return gated


def hard_gate(samples, probs, thr=0.5) -> list[dict]:
    gated = []
    for sample, prob in zip(samples, probs):
        if len(sample["pred_scores"]) == 0:
            gated.append(sample)
            continue
        keep = prob[sample["pred_labels"]] >= thr
        gated.append(
            {
                **sample,
                "pred_boxes": sample["pred_boxes"][keep],
                "pred_scores": sample["pred_scores"][keep],
                "pred_labels": sample["pred_labels"][keep],
            }
        )
    return gated


def gt_presence(records) -> torch.Tensor:
    y = torch.zeros(len(records), len(CLASSES))
    for i, rec in enumerate(records):
        for box in rec["boxes"]:
            y[i, int(box[4])] = 1
    return y


def temporal_summary(records, probs, target) -> dict:
    pred_b = probs >= 0.5
    tgt_b = target >= 0.5
    report = multilabel_report(probs, target, CLASSES)
    by_video: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        by_video.setdefault(rec["video"], []).append(i)
    seg, pred_flick, gt_flick = [], [], []
    for idxs in by_video.values():
        idxs = sorted(idxs, key=lambda i: records[i]["frame"])
        seg.append(segment_iou(pred_b[idxs], tgt_b[idxs]))
        pred_flick.append(flicker(pred_b[idxs]))
        gt_flick.append(flicker(tgt_b[idxs]))
    report["segment_iou"] = float(np.mean(seg)) if seg else None
    report["pred_flicker"] = float(np.mean(pred_flick)) if pred_flick else None
    report["gt_flicker"] = float(np.mean(gt_flick)) if gt_flick else None
    return report


def top_correct(sample) -> tuple[float, bool]:
    if len(sample["pred_scores"]) == 0:
        return 0.0, False
    i = int(sample["pred_scores"].argmax())
    score = float(sample["pred_scores"][i])
    box = sample["pred_boxes"][i]
    lab = int(sample["pred_labels"][i])
    ok = False
    for gt, gl in zip(sample["gt_boxes"], sample["gt_labels"]):
        if int(gl) != lab:
            continue
        iou = _iou(box, gt)
        ok = ok or iou >= 0.5
    return score, ok


def _iou(a, b) -> float:
    tl = torch.max(a[:2], b[:2])
    br = torch.min(a[2:], b[2:])
    wh = (br - tl).clamp(min=0)
    inter = float(wh[0] * wh[1])
    area_a = float((a[2] - a[0]).clamp(min=0) * (a[3] - a[1]).clamp(min=0))
    area_b = float((b[2] - b[0]).clamp(min=0) * (b[3] - b[1]).clamp(min=0))
    return inter / (area_a + area_b - inter + 1e-7)


def choose_threshold(samples, probs, cfg) -> dict:
    rows = []
    confs, correct = [], []
    for sample, prob in zip(samples, probs):
        score, ok = top_correct(sample)
        if len(sample["pred_scores"]):
            i = int(sample["pred_scores"].argmax())
            score = min(score, float(prob[int(sample["pred_labels"][i])]))
        confs.append(score)
        correct.append(ok)
    confs = np.asarray(confs)
    correct = np.asarray(correct)
    chosen = None
    for t in np.round(np.arange(0.1, 0.95, 0.05), 2):
        act = confs >= t
        n = int(act.sum())
        tp = int((act & correct).sum())
        precision = tp / n if n else 0.0
        recall = tp / max(int(correct.sum()), 1)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"threshold": float(t), "frames": n, "precision": precision, "recall": recall, "f1": f1})
        if chosen is None and precision >= cfg["uncertainty"]["min_precision"] and n >= cfg["uncertainty"]["min_frames"]:
            chosen = float(t)
    if chosen is None and rows:
        chosen = max(rows, key=lambda r: (r["f1"], -r["threshold"]))["threshold"]
    return {"threshold": chosen, "sweep": rows, "rule": "min(top detection score, temporal probability of that class)"}


def confusion(samples, score_thr=0.05, iou_thr=0.5) -> np.ndarray:
    mat = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for sample in samples:
        order = sample["pred_scores"].argsort(descending=True)
        used = set()
        for i in order.tolist():
            if float(sample["pred_scores"][i]) < score_thr:
                continue
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(sample["gt_boxes"]):
                if j in used:
                    continue
                iou = _iou(sample["pred_boxes"][i], gt)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thr and best_j >= 0:
                used.add(best_j)
                mat[int(sample["gt_labels"][best_j]), int(sample["pred_labels"][i])] += 1
    return mat


def _scale_boxes(boxes, sx: float, sy: float):
    if not torch.is_tensor(boxes) or boxes.numel() == 0:
        return boxes
    out = boxes.clone()
    out[:, [0, 2]] *= sx
    out[:, [1, 3]] *= sy
    return out


def failure_analysis(samples, model_hw: tuple[int, int]) -> dict:
    misses = []
    forceps = {"cadiere forceps", "bipolar forceps", "force bipolar", "needle driver"}
    border = small = motion_miss = rare_miss = forceps_wrong = 0
    border_n = small_n = 0
    blur_hit, blur_miss = [], []
    motion_hit, motion_miss_vals = [], []
    prev = {}
    examples = []
    for sample in samples:
        img = cv2.imread(sample["file"])
        if img is None:
            continue
        sx = img.shape[1] / model_hw[1]
        sy = img.shape[0] / model_hw[0]
        sample = {
            **sample,
            "gt_boxes": _scale_boxes(sample["gt_boxes"], sx, sy),
            "pred_boxes": _scale_boxes(sample["pred_boxes"], sx, sy),
        }
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        prev_img = prev.get(sample["video"])
        motion = None
        if prev_img is not None and prev_img[0] + 1 == sample["frame"]:
            motion = float(np.mean(np.abs(gray.astype(np.float32) - prev_img[1])))
        prev[sample["video"]] = (sample["frame"], gray)
        matched_gt = set()
        order = sample["pred_scores"].argsort(descending=True)
        pairs = []
        for i in order.tolist():
            if float(sample["pred_scores"][i]) < 0.05:
                continue
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(sample["gt_boxes"]):
                if j in matched_gt:
                    continue
                iou = _iou(sample["pred_boxes"][i], gt)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= 0.5 and best_j >= 0:
                matched_gt.add(best_j)
                gt_name = CLASSES[int(sample["gt_labels"][best_j])]
                pr_name = CLASSES[int(sample["pred_labels"][i])]
                pairs.append((gt_name, pr_name, best_iou))
                if gt_name in forceps and pr_name in forceps and gt_name != pr_name:
                    forceps_wrong += 1
                    examples.append((sample, "forceps_confusion", i))
        for j, gt in enumerate(sample["gt_boxes"]):
            x1, y1, x2, y2 = [float(v) for v in gt]
            w, h = x2 - x1, y2 - y1
            touches = x1 <= 2 or y1 <= 2 or x2 >= img.shape[1] - 2 or y2 >= img.shape[0] - 2
            is_small = w < 32 or h < 32
            border_n += int(touches)
            small_n += int(is_small)
            hit = j in matched_gt
            (blur_hit if hit else blur_miss).append(blur)
            if motion is not None:
                (motion_hit if hit else motion_miss_vals).append(motion)
            if hit:
                continue
            name = CLASSES[int(sample["gt_labels"][j])]
            if touches:
                border += 1
                examples.append((sample, "border", j))
            if is_small:
                small += 1
                examples.append((sample, "small", j))
            if motion is not None and motion > 15:
                motion_miss += 1
                examples.append((sample, "motion", j))
            if name in {"vessel sealer", "grasping retractor", "clip applier"}:
                rare_miss += 1
                examples.append((sample, "rare", j))
            misses.append({"image": sample["image_id"], "class": name, "blur": blur, "motion": motion, "border": touches, "small": is_small})
    saved = _save_examples(examples)
    def med(xs):
        return float(np.median(xs)) if xs else None
    return {
        "missed_boxes": len(misses),
        "border_misses": border,
        "border_boxes": border_n,
        "small_misses": small,
        "small_boxes": small_n,
        "motion_misses_diff_gt_15": motion_miss,
        "forceps_family_confusions": forceps_wrong,
        "rare_class_misses": rare_miss,
        "median_blur_hit": med(blur_hit),
        "median_blur_miss": med(blur_miss),
        "median_motion_hit": med(motion_hit),
        "median_motion_miss": med(motion_miss_vals),
        "example_images": saved,
    }


def _save_examples(examples) -> list[str]:
    dest = OUT / "failure_cases"
    dest.mkdir(parents=True, exist_ok=True)
    saved, seen = [], set()
    for sample, kind, _ in examples:
        if kind in seen:
            continue
        img = cv2.imread(sample["file"])
        if img is None:
            continue
        for box, lab in zip(sample["gt_boxes"], sample["gt_labels"]):
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 180, 0), 2)
            cv2.putText(img, CLASSES[int(lab)], (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 0), 1)
        for box, lab, score in zip(sample["pred_boxes"], sample["pred_labels"], sample["pred_scores"]):
            if float(score) < 0.2:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 220), 2)
            cv2.putText(img, f"{CLASSES[int(lab)]} {float(score):.2f}", (x1, min(img.shape[0] - 4, y2 + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 220), 1)
        path = dest / f"{kind}_{sample['video']}_{sample['frame']}.jpg"
        cv2.imwrite(str(path), img)
        saved.append(str(path))
        seen.add(kind)
        if len(seen) >= 6:
            break
    return saved


def plot_confusion(mat: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), [c.replace(" ", "\n") for c in CLASSES], fontsize=7)
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title("Matched detections at IoU 0.5")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_sweep(rows, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["threshold"] for r in rows], [r["precision"] for r in rows], label="precision")
    ax.plot([r["threshold"] for r in rows], [r["recall"] for r in rows], label="recall")
    ax.set_xlabel("confidence threshold")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Validation operating points")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_report(results: dict, path: Path) -> None:
    d = results["detectors"]
    lines = [
        "# SurgiSense evaluation",
        "",
        "Test clip is video 7. Validation clip is video 4. Metrics below are computed from the trained checkpoints.",
        "",
        "## Detection",
        "",
        "| model | val mAP@0.5 | test mAP@0.5 | test mAP@0.5:0.95 | test F1 @ score 0.5 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, row in d.items():
        lines.append(
            f"| {name} | {fmt(row['val_map50'])} | {fmt(row['test']['map50'])} | {fmt(row['test']['map50_95'])} | {fmt(row['test']['f1_at_0.5'])} |"
        )
    lines += [
        "",
        "## Temporal presence",
        "",
        "| model | val macro-F1 | test macro-F1 | test segment IoU | pred flicker | gt flicker |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, row in results["presence"].items():
        lines.append(
            f"| {name} | {fmt(row.get('val_macro_f1'))} | {fmt(row['test']['macro_f1'])} | {fmt(row['test'].get('segment_iou'))} | {fmt(row['test'].get('pred_flicker'))} | {fmt(row['test'].get('gt_flicker'))} |"
        )
    g = results["gating"]
    lines += [
        "",
        "## Temporal gating of the selected detector",
        "",
        f"Selected detector: `{results['selected_detector']}`.",
        "",
        f"- ungated test mAP@0.5: {fmt(g['ungated_map50'])}",
        f"- score × temporal probability, test mAP@0.5: {fmt(g['soft_map50'])}",
        f"- drop boxes whose temporal probability is below 0.5, test mAP@0.5: {fmt(g['hard_map50'])}",
        "",
        "## Uncertainty",
        "",
        f"Threshold selected on validation: {fmt(results['uncertainty']['threshold'])}.",
        "A frame is uncertain when min(top-box score, temporal probability of that class) is below the threshold. The assistant holds in that state.",
        "",
        "## Error analysis",
        "",
        "```json",
        json.dumps(results["failures"], indent=2),
        "```",
        "",
        "Green boxes in `evaluation/failure_cases/` are ground truth. Red boxes are predictions with score at least 0.2.",
        "",
        "This is a research and simulation prototype. It is not clinically validated and it does not perform surgery.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Evaluate SurgiSense checkpoints")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--index", default="data/processed/index.json")
    args = p.parse_args()
    cfg = load_config(args.config)
    dev = device()
    index = load_index(args.index)
    splits = {name: records_for_split(index, name) for name in ("val", "test")}
    feat_path = Path("data/processed/features.pt")
    if not feat_path.exists():
        raise SystemExit("missing data/processed/features.pt. Run train_temporal first.")
    feats = torch.load(feat_path, weights_only=False)
    OUT.mkdir(exist_ok=True)
    (OUT / "predictions").mkdir(exist_ok=True)

    detectors = {}
    collected = {}
    for tag in ("baseline", "aug"):
        model, ckpt = load_detector(Path(f"checkpoints/detector_{tag}_best.pt"), dev)
        collected[tag] = {split: collect(model, splits[split], dev, cfg) for split in ("val", "test")}
        detectors[tag] = {
            "val_map50": ckpt.get("best_map50"),
            "augment": ckpt.get("augment"),
            "test": detection_metrics(collected[tag]["test"], len(CLASSES)),
            "val": detection_metrics(collected[tag]["val"], len(CLASSES)),
        }
        print(f"{tag} test mAP@0.5 {detectors[tag]['test']['map50']:.3f}")

    selected = max(detectors, key=lambda tag: detectors[tag]["val_map50"] or -1)
    presence = {}
    probs = {}
    for name, path in (("frame", "checkpoints/temporal_frame.pt"), ("best", "checkpoints/temporal_best.pt")):
        model, ckpt = load_temporal(Path(path), dev)
        for split in ("val", "test"):
            probs[(name, split)] = presence_probs(model, ckpt, splits[split], feats[split], dev)
        test_report = temporal_summary(splits["test"], probs[(name, "test")], gt_presence(splits["test"]))
        val_report = multilabel_report(probs[(name, "val")], gt_presence(splits["val"]), CLASSES)
        presence[name] = {"val_macro_f1": val_report["macro_f1"], "test": test_report, "kind": ckpt["kind"], "window": ckpt.get("window")}
        print(f"presence {name} test macro-F1 {test_report['macro_f1']:.3f}")

    base_samples = collected[selected]["test"]
    soft = rescale(base_samples, probs[("best", "test")])
    hard = hard_gate(base_samples, probs[("best", "test")])
    gating = {
        "ungated_map50": detection_metrics(base_samples, len(CLASSES))["map50"],
        "soft_map50": detection_metrics(soft, len(CLASSES))["map50"],
        "hard_map50": detection_metrics(hard, len(CLASSES))["map50"],
    }
    unc = choose_threshold(collected[selected]["val"], probs[("best", "val")], cfg)
    (OUT / "threshold.json").write_text(json.dumps(_clean(unc), indent=2))
    mat = confusion(base_samples)
    plot_confusion(mat, OUT / "confusion_matrix.png")
    plot_sweep(unc["sweep"], OUT / "threshold_sweep.png")
    failures = failure_analysis(base_samples, (cfg["image_height"], cfg["image_width"]))
    # Hold/track counts on the test clip at the chosen threshold.
    holds = tracks = 0
    prev_uncertain = True
    history = []
    for sample, prob in zip(base_samples, probs[("best", "test")]):
        tools = []
        for box, score, lab in zip(sample["pred_boxes"], sample["pred_scores"], sample["pred_labels"]):
            if float(score) < 0.05:
                continue
            name = CLASSES[int(lab)]
            tools.append({"name": name, "score": min(float(score), float(prob[int(lab)])), "box": [float(v) for v in box]})
        presence_map = {CLASSES[c]: float(prob[c]) for c in range(len(CLASSES))}
        state = build_state(0.0, sample["video"], sample["frame"], tools, presence_map, unc["threshold"], history, prev_uncertain)
        cmd = assistant_command(state.to_dict(), unc["threshold"])
        prev_uncertain = state.uncertain
        history = state.history
        holds += int(cmd["mode"] == "hold")
        tracks += int(cmd["mode"] == "track")
    results = {
        "hardware": hardware(),
        "selected_detector": selected,
        "detectors": detectors,
        "presence": presence,
        "gating": gating,
        "uncertainty": unc,
        "assistant_on_test": {"hold_frames": holds, "track_frames": tracks},
        "failures": failures,
        "confusion_matrix": mat.tolist(),
    }
    (OUT / "results.json").write_text(json.dumps(_clean(results), indent=2))
    write_report(_clean(results), OUT / "report.md")
    print(f"wrote {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
