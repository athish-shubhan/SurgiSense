"""Dataset figures: class counts, clip lengths, split sizes."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from surgisense.data import CLASSES, load_index


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Plot SurgVU public-clip statistics")
    p.add_argument("--index", default="data/processed/index.json")
    p.add_argument("--out", default="evaluation/figures")
    args = p.parse_args()
    index = load_index(args.index)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    split_counts = Counter()
    for rec in index["frames"]:
        split_counts[rec["split"]] += 1
        for box in rec["boxes"]:
            counts[CLASSES[int(box[4])]] += 1
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(CLASSES)
    ax.barh(names, [counts[n] for n in names], color="#2c5f2d")
    ax.set_xlabel("boxes")
    ax.set_title("Tool boxes in the public SurgVU clips")
    fig.tight_layout()
    fig.savefig(out / "class_counts.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    order = ["train", "val", "test"]
    ax.bar(order, [split_counts[s] for s in order], color=["#1f4e79", "#bc6c25", "#9b2226"])
    ax.set_ylabel("frames")
    ax.set_title("Video-level split")
    fig.tight_layout()
    fig.savefig(out / "split_counts.png", dpi=120)
    plt.close(fig)

    videos = index.get("videos") or []
    if videos:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        labels = [f"{v['video']} ({v['split']})" for v in videos]
        ax.bar(labels, [v["duration_s"] or 0 for v in videos], color="#335c67")
        ax.set_ylabel("seconds at 1 fps")
        ax.set_title("Clip duration")
        ax.tick_params(axis="x", labelrotation=45)
        fig.tight_layout()
        fig.savefig(out / "clip_duration.png", dpi=120)
        plt.close(fig)
    stats = {
        "frames": len(index["frames"]),
        "boxes": sum(counts.values()),
        "class_counts": dict(counts),
        "split_frames": dict(split_counts),
        "videos": videos,
        "corrupt_frames": index.get("corrupt_frames", 0),
    }
    (out / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({"frames": stats["frames"], "boxes": stats["boxes"], "splits": stats["split_frames"]}))


if __name__ == "__main__":
    main()
