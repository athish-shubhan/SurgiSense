"""Animated replay of an inference log: boxes, confidence against the hold threshold, tool presence.

    python -m surgisense replay            # writes evaluation/replay.gif from evaluation/inference/predictions.jsonl

The log holds everything drawn here. The endoscope frames themselves are not
shown: the SurgVU terms do not allow redistributing them, so each box is drawn
on an empty 640x512 frame.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

MIN_BOX_SCORE = 0.15
FRAME_W, FRAME_H = 640, 512
COLOURS = ["#4cc9f0", "#f72585", "#ffd166", "#06d6a0", "#b388ff", "#ff9f1c", "#90be6d", "#e63946"]
BG, PANEL, INK, MUTED = "#0d1117", "#161b22", "#e6edf3", "#8b949e"
TRACK, HOLD = "#2ea043", "#d29922"


def load_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def frame_view(record: dict, min_score: float = MIN_BOX_SCORE) -> dict:
    """The parts of one log record that the replay draws."""
    state, command = record["state"], record["command"]
    boxes = [t for t in state["tools"] if t["score"] >= min_score]
    return {
        "frame": state["frame"],
        "confidence": float(state["confidence"]),
        "mode": command["mode"],
        "tool": command.get("tool"),
        "boxes": sorted(boxes, key=lambda t: t["score"]),  # strongest drawn last, on top
        "presence": state.get("presence", {}),
    }


def tool_colours(records: list[dict]) -> dict[str, str]:
    names = sorted({t["name"] for r in records for t in r["state"]["tools"]} | {n for r in records for n in r["state"].get("presence", {})})
    return {name: COLOURS[i % len(COLOURS)] for i, name in enumerate(names)}


def render(records: list[dict], threshold: float, start: int, count: int, out: Path, fps: int = 12, history: int = 90) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    views = [frame_view(r) for r in records]
    colours = tool_colours(records)
    names = sorted(colours)
    total = len(views)
    stop = min(total, start + count)

    fig = plt.figure(figsize=(9.6, 5.2), dpi=96, facecolor=BG)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.18, 1], height_ratios=[1, 1.05], left=0.035, right=0.975, top=0.86, bottom=0.09,
                            wspace=0.16, hspace=0.42)
    ax_frame, ax_conf, ax_pres = fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])
    for ax in (ax_frame, ax_conf, ax_pres):
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.tick_params(colors=MUTED, labelsize=8)

    images = []
    for i in range(start, stop):
        v = views[i]
        tracking = v["mode"] == "track"

        ax_frame.clear()
        ax_frame.set_facecolor("#0a0d12")
        ax_frame.set_xlim(0, FRAME_W)
        ax_frame.set_ylim(FRAME_H, 0)
        ax_frame.set_xticks([])
        ax_frame.set_yticks([])
        ax_frame.set_title(f"Detections, test clip frame {v['frame']} of {total} (frames not shown: dataset terms)",
                           color=MUTED, fontsize=9, loc="left")
        for t in v["boxes"]:
            x0, y0, x1, y1 = t["box"]
            c = colours[t["name"]]
            alpha = 0.35 + 0.65 * min(1.0, t["score"] / 0.7)
            ax_frame.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=True, facecolor=c, alpha=alpha * 0.16, linewidth=0))
            ax_frame.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=c, alpha=alpha, linewidth=2))
            ax_frame.text(x0, max(14, y0 - 6), f"{t['name']}  {t['score']:.2f}", color=c, alpha=alpha, fontsize=8.5, fontweight="bold")

        lo = max(0, i - history + 1)
        xs = list(range(lo, i + 1))
        conf = [views[k]["confidence"] for k in xs]
        ax_conf.clear()
        ax_conf.set_facecolor(PANEL)
        ax_conf.fill_between(xs, 0, 1, where=[views[k]["mode"] == "hold" for k in xs], step="mid", color=HOLD, alpha=0.16, linewidth=0)
        ax_conf.plot(xs, conf, color=INK, linewidth=1.4)
        ax_conf.axhline(threshold, color=HOLD, linestyle="--", linewidth=1.1)
        ax_conf.scatter([i], [v["confidence"]], color=TRACK if tracking else HOLD, s=34, zorder=5)
        ax_conf.set_xlim(i - history + 1, i + 3)
        ax_conf.set_ylim(0, 1)
        ax_conf.set_title(f"Confidence vs hold threshold ({threshold:.2f}); shaded = arm holding", color=MUTED, fontsize=9, loc="left")
        ax_conf.tick_params(colors=MUTED, labelsize=8)

        ax_pres.clear()
        ax_pres.set_facecolor(PANEL)
        values = [float(v["presence"].get(n, 0.0)) for n in names]
        ax_pres.barh(range(len(names)), values, color=[colours[n] for n in names], height=0.62)
        ax_pres.set_yticks(range(len(names)))
        ax_pres.set_yticklabels(names, color=INK, fontsize=8)
        ax_pres.set_xlim(0, 1)
        ax_pres.invert_yaxis()
        ax_pres.set_title("Tool presence over the last 16 frames (temporal model)", color=MUTED, fontsize=9, loc="left")
        ax_pres.tick_params(colors=MUTED, labelsize=8)

        for old in fig.texts[:]:
            old.remove()
        fig.text(0.035, 0.93, "SurgiSense replay", color=INK, fontsize=15, fontweight="bold")
        label = f"TRACK  {v['tool']}" if tracking else f"HOLD  confidence {v['confidence']:.2f} is below {threshold:.2f}"
        fig.text(0.975, 0.93, label, color="#0d1117", fontsize=11, fontweight="bold", ha="right",
                 bbox={"boxstyle": "round,pad=0.45", "facecolor": TRACK if tracking else HOLD, "edgecolor": "none"})

        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=BG)
        buf.seek(0)
        images.append(Image.open(buf).convert("RGB").quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    plt.close(fig)

    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return len(images)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=Path("evaluation/inference/predictions.jsonl"))
    ap.add_argument("--threshold-file", type=Path, default=Path("evaluation/threshold.json"))
    ap.add_argument("--out", type=Path, default=Path("evaluation/replay.gif"))
    ap.add_argument("--start", type=int, default=290, help="first frame of the replayed segment")
    ap.add_argument("--frames", type=int, default=230)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    records = load_log(args.log)
    threshold = float(json.loads(args.threshold_file.read_text())["threshold"])
    n = render(records, threshold, args.start, args.frames, args.out, args.fps)
    print(f"wrote {args.out}: {n} frames, {args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
