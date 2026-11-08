import json
from pathlib import Path

import numpy as np
import torch
import yaml

from surgisense.context import assistant_command, build_state, events_from
from surgisense.data import (
    CLASSES,
    SOURCE_TO_CONTIGUOUS,
    assert_disjoint_splits,
    coco_to_records,
    flip_boxes,
    parse_task_csv,
    parse_tool_csv,
)
from surgisense.detector import ToolDetector, detection_loss, predict
from surgisense.metrics import detection_metrics
from surgisense.sim import ArmSim, integrate


def test_disjoint_splits_reject_leakage():
    try:
        assert_disjoint_splits({"train": ["1"], "val": ["1"], "test": ["2"]})
    except ValueError:
        return
    raise AssertionError("shared video was accepted")


def test_csv_labels():
    tasks = parse_task_csv("index,start_part,start_time,stop_part,stop_time,groundtruth_taskname\n0,1,0,1,10,Suturing\n")
    assert tasks[0]["groundtruth_taskname"] == "Suturing"
    tools = parse_tool_csv(
        "index,install_case_part,install_case_time,uninstall_case_part,uninstall_case_time,arm,commercial_toolname,groundtruth_toolname\n"
        "0,1,00:01:00,1,00:02:00,USM1,Large Needle Driver,needle driver\n"
    )
    assert tools[0]["groundtruth_toolname"] == "needle driver"


def test_coco_records_and_split_boxes():
    coco = {
        "images": [{"id": 3, "width": 640, "height": 512, "file_name": "0000000003.jpg"}],
        "annotations": [
            {"id": 1, "image_id": 3, "category_id": 11, "bbox": [10, 20, 30, 40]},
            {"id": 2, "image_id": 3, "category_id": 99, "bbox": [0, 0, 10, 10]},
        ],
        "categories": [],
    }
    records, skipped = coco_to_records(coco, "7", "test")
    assert len(records) == 1
    assert records[0]["frame"] == 3
    assert records[0]["boxes"][0][4] == SOURCE_TO_CONTIGUOUS[11]
    assert skipped
    boxes = torch.tensor([[10.0, 5.0, 30.0, 25.0]])
    flipped = flip_boxes(boxes, 100)
    assert torch.allclose(flipped[0], torch.tensor([70.0, 5.0, 90.0, 25.0]))


def test_detector_forward_loss_and_checkpoint(tmp_path: Path):
    model = ToolDetector(len(CLASSES), pretrained=False)
    images = torch.rand(2, 3, 128, 128)
    pred = model(images)
    assert pred.shape == (2, 1 + len(CLASSES) + 4, 8, 8)
    batch = [
        {"boxes": torch.tensor([[16.0, 16.0, 48.0, 48.0]]), "labels": torch.tensor([3])},
        {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)},
    ]
    losses = detection_loss(pred, batch, stride=16)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    path = tmp_path / "det.pt"
    torch.save({"model": model.state_dict(), "classes": CLASSES}, path)
    again = ToolDetector(len(CLASSES), pretrained=False)
    again.load_state_dict(torch.load(path, weights_only=False)["model"])
    again.eval()
    dets = predict(again(images), (128, 128))
    assert len(dets) == 2
    assert dets[0]["boxes"].ndim == 2 and dets[0]["scores"].ndim == 1


def test_map_perfect_and_miss():
    box = torch.tensor([[8.0, 8.0, 40.0, 30.0]])
    perfect = {
        "image_id": "v:0",
        "pred_boxes": box,
        "pred_scores": torch.tensor([0.9]),
        "pred_labels": torch.tensor([1]),
        "gt_boxes": box.clone(),
        "gt_labels": torch.tensor([1]),
    }
    hit = detection_metrics([perfect], 3)
    assert hit["map50"] > 0.99
    miss = dict(perfect)
    miss["pred_boxes"] = torch.tensor([[80.0, 80.0, 100.0, 100.0]])
    assert detection_metrics([miss], 3)["map50"] == 0.0


def test_uncertainty_holds_and_tracks():
    low = build_state(
        0.0, "7", 1,
        [{"name": "needle driver", "score": 0.2, "box": [0, 0, 1, 1]}],
        {"needle driver": 0.9},
        0.5,
    )
    assert low.uncertain
    assert assistant_command(low.to_dict(), 0.5)["mode"] == "hold"
    high = build_state(
        1.0, "7", 2,
        [{"name": "needle driver", "score": 0.8, "box": [0, 0, 1, 1]}],
        {"needle driver": 0.7},
        0.5,
        history=[["needle driver"]],
        prev_uncertain=True,
    )
    assert not high.uncertain
    assert abs(high.confidence - 0.7) < 1e-6
    cmd = assistant_command(high.to_dict(), 0.5)
    assert cmd["mode"] == "track" and cmd["joints"] is not None
    assert "became_confident" in high.events
    payload = json.loads(json.dumps(high.to_dict()))
    assert payload["tools"][0]["name"] == "needle driver"
    assert "entered:cadiere forceps" in events_from(set(), {"cadiere forceps"}, False, False)


def test_hold_does_not_move_arm():
    q = np.zeros(2)
    assert np.allclose(integrate(q, None), q)
    moved = integrate(q, np.array([1.0, 0.0]), gain=0.5)
    assert moved[0] == 0.5 and moved[1] == 0.0
    sim = ArmSim(backend="mujoco")
    tracked = sim.step([0.6, 0.7])
    held = sim.step(None)
    assert sim.backend == "mujoco"
    assert np.allclose(held, tracked)


def test_config_roundtrip(tmp_path: Path):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"splits": {"train": ["1"], "val": ["2"], "test": ["3"]}}))
    from surgisense.data import load_config

    cfg = load_config(path)
    assert_disjoint_splits(cfg["splits"])


def test_watchdog_rejects_reordered_and_detects_stale():
    from surgisense.context import Watchdog

    wd = Watchdog(timeout_s=0.25)
    assert wd.stale(0.0)  # nothing received yet
    assert wd.accept("7", 10, now=1.0)
    assert not wd.stale(1.2)
    assert wd.stale(1.3)
    assert not wd.accept("7", 9, now=1.3)  # reordered
    assert not wd.accept("7", 10, now=1.3)  # duplicate
    assert wd.rejected == 2
    assert wd.accept("8", 0, now=1.4)  # a new clip restarts the frame count
    assert not wd.stale(1.5)


def test_arm_holds_position_across_control_ticks():
    sim = ArmSim(backend="mujoco")
    for _ in range(20):
        sim.step([0.6, 0.7], n=2)
    q = sim.step(None, n=2)
    for _ in range(50):
        assert np.array_equal(sim.step(None, n=2), q)


def test_publish_order_swaps_pairs():
    import importlib.util

    spec = importlib.util.spec_from_file_location("perception_node", Path(__file__).parents[1] / "ros2/perception_node.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.publish_order(6, 0) == [0, 1, 2, 3, 4, 5]
    assert mod.publish_order(6, 2) == [0, 1, 3, 2, 5, 4]
