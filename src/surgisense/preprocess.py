"""Extract annotated frames and write a video-level index."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from surgisense.data import CLASSES, assert_disjoint_splits, coco_to_records, load_config


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Preprocess SurgVU public clips")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--raw", default="data/raw/cat1")
    p.add_argument("--out", default="data/processed")
    args = p.parse_args()
    cfg = load_config(args.config)
    assert_disjoint_splits(cfg["splits"])
    split_of = {}
    for name, videos in cfg["splits"].items():
        for video in videos:
            split_of[str(video)] = name

    raw = Path(args.raw)
    out = Path(args.out)
    frame_root = out / "frames"
    frame_root.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        raise SystemExit(
            f"missing {raw}. Run: python -m surgisense.download\n"
            "The public tool clips are a separate download from the 344 GB video archive."
        )

    videos = sorted(raw.glob("*_fps1.mp4"), key=lambda pth: int(pth.name.split("_")[0]))
    if not videos:
        raise SystemExit(f"no *_fps1.mp4 files in {raw}")

    frames = []
    skipped = []
    video_stats = []
    corrupt = 0
    for path in videos:
        video = path.name.split("_")[0]
        if video not in split_of:
            raise SystemExit(f"video {video} is not assigned a split in {args.config}")
        coco_path = raw / f"{video}_fps1_coco.json"
        if not coco_path.exists():
            raise SystemExit(f"missing annotations: {coco_path}")
        coco = json.loads(coco_path.read_text())
        records, notes = coco_to_records(coco, video, split_of[video])
        skipped.extend(notes)
        wanted = {r["frame"]: r for r in records}
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise SystemExit(f"could not open {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        saved = 0
        idx = 0
        dest_dir = frame_root / video
        dest_dir.mkdir(parents=True, exist_ok=True)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rec = wanted.get(idx)
            if rec is not None:
                dest = dest_dir / f"{idx:06d}.jpg"
                if frame is None or frame.size == 0:
                    corrupt += 1
                    skipped.append(f"video {video} frame {idx} empty")
                else:
                    if not dest.exists():
                        if not cv2.imwrite(str(dest), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
                            corrupt += 1
                            skipped.append(f"video {video} frame {idx} write failed")
                            idx += 1
                            continue
                    rec = dict(rec)
                    rec["file"] = str(dest)
                    frames.append(rec)
                    saved += 1
            idx += 1
        cap.release()
        missing = sorted(set(wanted) - {r["frame"] for r in frames if r["video"] == video})
        video_stats.append(
            {
                "video": video,
                "split": split_of[video],
                "fps": fps,
                "container_frames": nframes,
                "decoded_frames": idx,
                "annotated_frames": len(wanted),
                "saved_frames": saved,
                "missing_annotated_frames": missing,
                "width": width,
                "height": height,
                "duration_s": idx / fps if fps else None,
            }
        )
        print(f"video {video} split={split_of[video]} saved={saved} decoded={idx}")

    frames.sort(key=lambda r: (int(r["video"]), r["frame"]))
    index = {
        "dataset": "surgvu-public-tool-clips",
        "classes": CLASSES,
        "frames": frames,
        "videos": video_stats,
        "skipped_annotations": skipped,
        "corrupt_frames": corrupt,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.json").write_text(json.dumps(index))
    summary = {
        "frames": len(frames),
        "by_split": {
            name: sum(1 for r in frames if r["split"] == name) for name in ("train", "val", "test")
        },
        "corrupt_frames": corrupt,
        "skipped_annotations": len(skipped),
        "videos": video_stats,
    }
    (out / "stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["by_split"]))
    if corrupt:
        print(f"warning: {corrupt} corrupt frames skipped")


if __name__ == "__main__":
    main()
