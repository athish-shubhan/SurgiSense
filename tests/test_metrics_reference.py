"""detection_metrics against the official COCO evaluator on random scenes."""

import contextlib
import io

import numpy as np
import pytest
import torch

from surgisense.metrics import detection_metrics

pycocotools = pytest.importorskip("pycocotools")
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

NUM_CLASSES = 4


def random_scene(rng: np.random.Generator, n_images: int) -> list[dict]:
    samples = []
    for i in range(n_images):
        n_gt = int(rng.integers(0, 4))
        xy = rng.uniform(0, 200, size=(n_gt, 2))
        wh = rng.uniform(10, 80, size=(n_gt, 2))
        gt = np.concatenate([xy, xy + wh], 1)
        gt_lab = rng.integers(0, NUM_CLASSES, n_gt)
        preds, labs = [], []
        for box, lab in zip(gt, gt_lab):
            # Jittered copies: some match at 0.5 only, some at 0.95, some not at all.
            for _ in range(int(rng.integers(0, 3))):
                preds.append(box + rng.normal(0, rng.choice([1.0, 6.0, 20.0]), 4))
                labs.append(lab if rng.random() < 0.8 else rng.integers(0, NUM_CLASSES))
        for _ in range(int(rng.integers(0, 3))):  # pure false positives
            p = rng.uniform(0, 200, 2)
            preds.append(np.concatenate([p, p + rng.uniform(10, 80, 2)]))
            labs.append(rng.integers(0, NUM_CLASSES))
        pred = np.asarray(preds, dtype=np.float64).reshape(-1, 4)
        pred[:, 2:] = np.maximum(pred[:, 2:], pred[:, :2] + 1)
        samples.append(
            {
                "image_id": str(i),
                "pred_boxes": torch.tensor(pred, dtype=torch.float32),
                # Distinct scores: COCO and this code break score ties differently.
                "pred_scores": torch.tensor(rng.permutation(len(pred)) / 100 + rng.uniform(0.01, 0.5, len(pred)), dtype=torch.float32),
                "pred_labels": torch.tensor(np.asarray(labs, dtype=np.int64)),
                "gt_boxes": torch.tensor(gt, dtype=torch.float32).reshape(-1, 4),
                "gt_labels": torch.tensor(gt_lab, dtype=torch.long),
            }
        )
    return samples


def coco_map(samples: list[dict]) -> tuple[float, float]:
    images, anns, dets = [], [], []
    for i, s in enumerate(samples):
        images.append({"id": i, "width": 400, "height": 400})
        for b, lab in zip(s["gt_boxes"].tolist(), s["gt_labels"].tolist()):
            w, h = b[2] - b[0], b[3] - b[1]
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": lab, "bbox": [b[0], b[1], w, h], "area": w * h, "iscrowd": 0})
        for b, sc, lab in zip(s["pred_boxes"].tolist(), s["pred_scores"].tolist(), s["pred_labels"].tolist()):
            dets.append({"image_id": i, "category_id": lab, "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": sc})
    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": c} for c in range(NUM_CLASSES)]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        ev = COCOeval(gt, gt.loadRes(dets), "bbox")
        ev.params.maxDets = [1, 10, 100]
        ev.params.areaRng = [[0, 1e10]] * 4  # no area buckets
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return float(ev.stats[1]), float(ev.stats[0])


@pytest.mark.parametrize("seed", range(5))
def test_map_matches_pycocotools(seed):
    samples = random_scene(np.random.default_rng(seed), n_images=60)
    ours = detection_metrics(samples, NUM_CLASSES)
    ref50, ref50_95 = coco_map(samples)
    assert ours["map50"] == pytest.approx(ref50, abs=1e-6)
    assert ours["map50_95"] == pytest.approx(ref50_95, abs=1e-6)
