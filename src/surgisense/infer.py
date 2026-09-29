"""Sequential inference on a clip, with measured latency."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from surgisense.context import assistant_command, build_state
from surgisense.data import CLASSES, IMAGENET_MEAN, IMAGENET_STD, load_config
from surgisense.detector import ToolDetector, predict
from surgisense.evaluate import load_temporal
from surgisense.experiment import device
from surgisense.temporal import FeatureEncoder


def _load_detector(path, dev):
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    model = ToolDetector(len(CLASSES), pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(dev).eval(), ckpt


def _tensor(frame_bgr: np.ndarray, image_size: tuple[int, int]) -> tuple[torch.Tensor, float, float]:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    src_h, src_w = rgb.shape[:2]
    dst_h, dst_w = image_size
    resized = cv2.resize(rgb, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1)
    mean = x.new_tensor(IMAGENET_MEAN)[:, None, None]
    std = x.new_tensor(IMAGENET_STD)[:, None, None]
    return (x - mean) / std, src_w / dst_w, src_h / dst_h


def main() -> None:
    import argparse
    import resource
    import platform

    p = argparse.ArgumentParser(description="Run sequential SurgiSense inference")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--video", default="data/raw/cat1/7_fps1.mp4")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--temporal", default="checkpoints/temporal_best.pt")
    p.add_argument("--out", default="evaluation/inference")
    p.add_argument("--max-frames", type=int, default=0)
    args = p.parse_args()
    cfg = load_config(args.config)
    checkpoint = args.checkpoint
    if not checkpoint:
        results_path = Path("evaluation/results.json")
        if results_path.exists():
            selected = json.loads(results_path.read_text()).get("selected_detector")
            if selected:
                checkpoint = f"checkpoints/detector_{selected}_best.pt"
        if not checkpoint:
            checkpoint = "checkpoints/detector_aug_best.pt"
    dev = device()
    if not Path(checkpoint).exists():
        raise SystemExit(f"missing detector checkpoint {checkpoint}")
    if not Path(args.video).exists():
        raise SystemExit(f"missing video {args.video}")
    thr_path = Path("evaluation/threshold.json")
    threshold = cfg["uncertainty"]["threshold"]
    if thr_path.exists():
        threshold = json.loads(thr_path.read_text())["threshold"]
    model, _ = _load_detector(checkpoint, dev)
    temporal, tckpt = load_temporal(Path(args.temporal), dev)
    encoder = FeatureEncoder(pretrained=True).to(dev)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video}")
    video_id = Path(args.video).name.split("_")[0]
    out = Path(args.out)
    frame_dir = out / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (width, height))
    times = []
    rows = []
    feat_hist: list[torch.Tensor] = []
    history, prev_uncertain = [], True
    window = int(tckpt.get("window") or 1)
    seen = 0
    warmup = 8
    while True:
        if args.max_frames and seen >= args.max_frames:
            break
        t_read = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        image, scale_x, scale_y = _tensor(frame, (cfg["image_height"], cfg["image_width"]))
        t_pre = time.perf_counter()
        with torch.no_grad():
            pred = model(image[None].to(dev))
            if dev.type == "mps":
                torch.mps.synchronize()
            t_det = time.perf_counter()
            det = predict(
                pred,
                (cfg["image_height"], cfg["image_width"]),
                score_thresh=0.05,
                nms_iou=cfg["nms_iou"],
                max_det=cfg["max_det"],
            )[0]
            if dev.type == "mps":
                torch.mps.synchronize()
            t_decode = time.perf_counter()
            if len(det["boxes"]):
                det["boxes"] = det["boxes"].clone()
                det["boxes"][:, [0, 2]] *= scale_x
                det["boxes"][:, [1, 3]] *= scale_y
            feat = encoder(image[None].to(dev)).cpu()[0]
            feat_hist.append(feat)
            if tckpt["kind"] == "frame":
                prob = torch.sigmoid(temporal(feat[None].to(dev)))[0].cpu()
            else:
                seq = feat_hist[-window:]
                if len(seq) < window:
                    seq = [seq[0]] * (window - len(seq)) + seq
                prob = torch.sigmoid(temporal(torch.stack(seq)[None].to(dev)))[0].cpu()
            if dev.type == "mps":
                torch.mps.synchronize()
        t_temporal = time.perf_counter()
        tools = []
        for box, score, lab in zip(det["boxes"], det["scores"], det["labels"]):
            name = CLASSES[int(lab)]
            tools.append(
                {
                    "name": name,
                    "score": min(float(score), float(prob[int(lab)])),
                    "box": [float(v) for v in box],
                }
            )
        presence = {CLASSES[c]: float(prob[c]) for c in range(len(CLASSES))}
        latency = {
            "preprocess_ms": (t_pre - t_read) * 1000,
            "detector_ms": (t_det - t_pre) * 1000,
            "post_ms": (t_decode - t_det) * 1000,
            "temporal_ms": (t_temporal - t_decode) * 1000,
        }
        latency["model_ms"] = latency["detector_ms"] + latency["temporal_ms"]
        latency["total_ms"] = latency["preprocess_ms"] + latency["detector_ms"] + latency["post_ms"] + latency["temporal_ms"]
        state = build_state(seen / 1.0, video_id, seen, tools, presence, threshold, history, prev_uncertain, latency)
        cmd = assistant_command(state.to_dict(), threshold)
        prev_uncertain = state.uncertain
        history = state.history
        if seen >= warmup:
            times.append(latency)
        vis = frame.copy()
        for tool in tools:
            if tool["score"] < 0.2:
                continue
            x1, y1, x2, y2 = [int(v) for v in tool["box"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 220), 2)
            cv2.putText(vis, f"{tool['name']} {tool['score']:.2f}", (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1)
        banner = f"{'HOLD' if cmd['mode']=='hold' else 'TRACK'} conf {state.confidence:.2f}"
        cv2.putText(vis, banner, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        writer.write(vis)
        if seen % 40 == 0:
            cv2.imwrite(str(frame_dir / f"{seen:06d}.jpg"), vis)
        rows.append({"state": state.to_dict(), "command": cmd})
        seen += 1
    cap.release()
    writer.release()
    if not times:
        raise SystemExit("video produced no timed frames")
    def avg(key):
        return float(np.mean([t[key] for t in times]))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        rss_mb = rss / (1024 * 1024)
    else:
        rss_mb = rss / 1024
    summary = {
        "video": args.video,
        "frames": seen,
        "timed_frames": len(times),
        "warmup_excluded": warmup,
        "checkpoint": checkpoint,
        "temporal": args.temporal,
        "threshold": threshold,
        "device": str(dev),
        "resolution": [height, width],
        "batch_size": 1,
        "preprocess_ms": avg("preprocess_ms"),
        "detector_ms": avg("detector_ms"),
        "temporal_ms": avg("temporal_ms"),
        "model_ms": avg("model_ms"),
        "post_ms": avg("post_ms"),
        "total_ms": avg("total_ms"),
        "fps": 1000.0 / avg("total_ms"),
        "max_rss_mb": rss_mb,
        "mps_allocated_mb": (torch.mps.current_allocated_memory() / (1024 * 1024)) if dev.type == "mps" else None,
        "hold_frames": sum(1 for r in rows if r["command"]["mode"] == "hold"),
        "track_frames": sum(1 for r in rows if r["command"]["mode"] == "track"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
