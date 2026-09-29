"""SurgVU public-clip loading, video-level splits, and frame dataset."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

# Contiguous ids used by the models. Source ids are the COCO category ids
# that actually have boxes in the public clip release.
CLASSES = [
    "grasping retractor",
    "cadiere forceps",
    "bipolar forceps",
    "force bipolar",
    "clip applier",
    "monopolar curved scissor",
    "vessel sealer",
    "needle driver",
]
SOURCE_CATEGORY_IDS = [0, 1, 2, 3, 4, 7, 8, 11]
SOURCE_TO_CONTIGUOUS = {src: i for i, src in enumerate(SOURCE_CATEGORY_IDS)}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return cfg


def assert_disjoint_splits(splits: dict) -> None:
    names = ("train", "val", "test")
    seen = {}
    for name in names:
        if name not in splits:
            raise ValueError(f"config splits missing {name}")
        for video in splits[name]:
            vid = str(video)
            if vid in seen:
                raise ValueError(f"video {vid} is in both {seen[vid]} and {name}")
            seen[vid] = name


def parse_task_csv(text: str) -> list[dict]:
    return _parse_csv(text, "groundtruth_taskname")


def parse_tool_csv(text: str) -> list[dict]:
    return _parse_csv(text, "groundtruth_toolname")


def _parse_csv(text: str, required: str) -> list[dict]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("empty label csv")
    header = lines[0].split(",")
    if required not in header:
        raise ValueError(f"label csv missing {required}")
    rows = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) != len(header):
            raise ValueError(f"label row has {len(parts)} fields, header has {len(header)}")
        rows.append(dict(zip(header, parts)))
    return rows


def coco_to_records(coco: dict, video: str, split: str) -> tuple[list[dict], list[str]]:
    """Turn one clip's COCO dict into frame records. Skips unknown tool ids."""
    id_to_image = {im["id"]: im for im in coco["images"]}
    boxes_for = {im["id"]: [] for im in coco["images"]}
    skipped = []
    for ann in coco["annotations"]:
        src = int(ann["category_id"])
        if src not in SOURCE_TO_CONTIGUOUS:
            skipped.append(f"video {video} ann {ann.get('id')} category {src}")
            continue
        image = id_to_image.get(ann["image_id"])
        if image is None:
            skipped.append(f"video {video} ann {ann.get('id')} missing image")
            continue
        x, y, w, h = [float(v) for v in ann["bbox"]]
        width, height = int(image["width"]), int(image["height"])
        x1 = min(max(x, 0.0), width - 1.0)
        y1 = min(max(y, 0.0), height - 1.0)
        x2 = min(max(x + w, 0.0), float(width))
        y2 = min(max(y + h, 0.0), float(height))
        if x2 - x1 < 2 or y2 - y1 < 2:
            skipped.append(f"video {video} ann {ann.get('id')} degenerate box")
            continue
        boxes_for[ann["image_id"]].append(
            [x1, y1, x2, y2, SOURCE_TO_CONTIGUOUS[src]]
        )
    records = []
    for image in coco["images"]:
        frame = int(Path(image["file_name"]).stem)
        records.append(
            {
                "video": str(video),
                "frame": frame,
                "split": split,
                "width": int(image["width"]),
                "height": int(image["height"]),
                "boxes": boxes_for[image["id"]],
            }
        )
    records.sort(key=lambda r: r["frame"])
    return records, skipped


def load_index(path: str | Path) -> dict:
    with open(path) as f:
        index = json.load(f)
    if index.get("classes") != CLASSES:
        raise ValueError("index class list does not match this code")
    return index


def records_for_split(index: dict, split: str) -> list[dict]:
    return [r for r in index["frames"] if r["split"] == split]


def _normalize(image: torch.Tensor) -> torch.Tensor:
    mean = image.new_tensor(IMAGENET_MEAN)[:, None, None]
    std = image.new_tensor(IMAGENET_STD)[:, None, None]
    return (image - mean) / std


def flip_boxes(boxes: torch.Tensor, width: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes
    out = boxes.clone()
    x1 = boxes[:, 0].clone()
    x2 = boxes[:, 2].clone()
    out[:, 0] = width - x2
    out[:, 2] = width - x1
    return out


def color_jitter(image: torch.Tensor) -> torch.Tensor:
    # image is float CHW in [0, 1]
    if random.random() < 0.8:
        bright = 1.0 + random.uniform(-0.25, 0.25)
        contrast = 1.0 + random.uniform(-0.25, 0.25)
        mean = image.mean(dim=(1, 2), keepdim=True)
        image = (image * bright - mean) * contrast + mean
    return image.clamp(0, 1)


class FrameDataset(Dataset):
    def __init__(self, records: list[dict], augment: bool = False, image_size: tuple[int, int] | None = None):
        self.records = records
        self.augment = augment
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        path = Path(rec["file"])
        try:
            img = Image.open(path).convert("RGB")
        except OSError as exc:
            raise RuntimeError(f"corrupted frame {path}: {exc}") from exc
        image = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        boxes = torch.tensor(rec["boxes"], dtype=torch.float32).reshape(-1, 5)
        labels = boxes[:, 4].to(torch.long) if len(boxes) else torch.zeros(0, dtype=torch.long)
        xyxy = boxes[:, :4] if len(boxes) else torch.zeros(0, 4)
        if self.image_size is not None and tuple(image.shape[-2:]) != tuple(self.image_size):
            src_h, src_w = image.shape[-2:]
            dst_h, dst_w = self.image_size
            image = torch.nn.functional.interpolate(
                image[None], size=(dst_h, dst_w), mode="bilinear", align_corners=False
            )[0]
            if len(xyxy):
                xyxy = xyxy.clone()
                xyxy[:, [0, 2]] *= dst_w / src_w
                xyxy[:, [1, 3]] *= dst_h / src_h
        if self.augment and random.random() < 0.5:
            image = image.flip(-1)
            xyxy = flip_boxes(xyxy, image.shape[-1])
        if self.augment:
            image = color_jitter(image)
        return {
            "image": _normalize(image),
            "boxes": xyxy,
            "labels": labels,
            "video": rec["video"],
            "frame": rec["frame"],
            "file": rec["file"],
            "presence": presence_vector(labels),
        }


def presence_vector(labels: torch.Tensor) -> torch.Tensor:
    vec = torch.zeros(len(CLASSES), dtype=torch.float32)
    if len(labels):
        vec[labels.unique()] = 1.0
    return vec


def collate(batch: list[dict]) -> tuple[torch.Tensor, list[dict]]:
    images = torch.stack([b["image"] for b in batch])
    return images, batch
