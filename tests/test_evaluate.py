"""The evaluation entry points, end to end on a few synthetic frames (no SurgVU data, no ImageNet download)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from surgisense import evaluate
from surgisense.data import CLASSES
from surgisense.detector import ToolDetector

CFG = {
    "image_height": 64, "image_width": 80, "stride": 16, "batch_size": 2,
    "score_thresh": 0.01, "nms_iou": 0.4, "max_det": 40,
    "temporal": {"batch_size": 4},
    "splits": {"train": ["1"], "val": ["2"], "test": ["3"]},
}


def _frames(root: Path) -> list[dict]:
    rng = np.random.default_rng(0)
    frames = []
    for video, split in (("2", "val"), ("3", "test")):
        for frame in range(3):
            path = root / "frames" / video / f"{frame:06d}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 255, (128, 160, 3), dtype=np.uint8)).save(path)
            frames.append({"video": video, "frame": frame, "split": split, "width": 160, "height": 128,
                           "boxes": [[20.0, 20.0, 60.0, 60.0, frame % len(CLASSES)]], "file": str(path)})
    return frames


def _checkpoints(ckpt_dir: Path) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    for tag, val_map in (("baseline", 0.25), ("aug", 0.3)):
        model = ToolDetector(len(CLASSES), pretrained=False)
        torch.save({"model": model.state_dict(), "best_map50": val_map, "augment": tag == "aug"},
                   ckpt_dir / f"detector_{tag}_best.pt")


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    frames = _frames(tmp_path)
    _checkpoints(tmp_path / "checkpoints")
    (tmp_path / "index.json").write_text(json.dumps({"classes": CLASSES, "frames": frames}))
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(CFG))
    monkeypatch.chdir(tmp_path)
    return tmp_path, {s: [f for f in frames if f["split"] == s] for s in ("val", "test")}


def test_score_detectors_scores_both_checkpoints(workspace):
    root, splits = workspace
    detectors, collected = evaluate.score_detectors(splits, CFG, torch.device("cpu"), root / "checkpoints")
    assert set(detectors) == {"baseline", "aug"}
    for tag, val_map in (("baseline", 0.25), ("aug", 0.3)):
        assert detectors[tag]["val_map50"] == val_map  # the value stored at training time, used for selection
        for split in ("val", "test"):
            assert 0.0 <= detectors[tag][split]["map50"] <= 1.0
            assert len(collected[tag][split]) == 3
            assert collected[tag][split][0]["image_id"] == f"{splits[split][0]['video']}:0"


def test_detectors_only_prints_metrics_and_writes_nothing(workspace, monkeypatch, capsys):
    root, _ = workspace
    monkeypatch.setattr(evaluate, "device", lambda: torch.device("cpu"))
    monkeypatch.setattr(sys, "argv", ["evaluate", "--config", "cfg.yaml", "--index", "index.json", "--detectors-only"])
    before = sorted(p.relative_to(root) for p in root.rglob("*"))
    evaluate.main()
    after = sorted(p.relative_to(root) for p in root.rglob("*"))
    assert after == before
    captured = capsys.readouterr()
    summary = json.loads(captured.out)  # stdout is only the JSON, so it can be piped to jq
    assert "baseline val mAP@0.5" in captured.err
    assert summary["device"] == "cpu"
    assert set(summary["detectors"]["aug"]) == {"val", "test"}
    assert set(summary["detectors"]["aug"]["test"]) == {"map50", "map50_95"}


def test_features_are_extracted_once_then_reused(tmp_path):
    splits = {"val": [{"x": 1}] * 2, "test": [{"x": 1}] * 3}
    calls = []

    def fake_extract(records, dev, batch_size, size):
        calls.append((len(records), batch_size, size))
        return torch.ones(len(records), 5)

    path = tmp_path / "processed" / "features.pt"
    feats = evaluate.load_or_extract_features(path, splits, "cpu", CFG, extract_fn=fake_extract)
    assert calls == [(2, 4, (64, 80)), (3, 4, (64, 80))]
    assert feats["test"].shape == (3, 5) and path.exists()

    again = evaluate.load_or_extract_features(path, splits, "cpu", CFG, extract_fn=fake_extract)
    assert len(calls) == 2  # cached, not re-extracted
    assert torch.equal(again["val"], feats["val"])
