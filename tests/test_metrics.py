"""Hand-checked cases for the metrics behind the README's mAP and macro-F1 numbers."""

import pytest
import torch

from surgisense.metrics import box_iou_matrix, detection_metrics, flicker, multilabel_report, precision_recall, segment_iou


def _sample(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, image_id="v:0"):
    return {
        "image_id": image_id,
        "pred_boxes": torch.tensor(pred_boxes, dtype=torch.float32).reshape(-1, 4),
        "pred_scores": torch.tensor(pred_scores, dtype=torch.float32),
        "pred_labels": torch.tensor(pred_labels, dtype=torch.long),
        "gt_boxes": torch.tensor(gt_boxes, dtype=torch.float32).reshape(-1, 4),
        "gt_labels": torch.tensor(gt_labels, dtype=torch.long),
    }


GT_A = [0.0, 0.0, 10.0, 10.0]
GT_B = [50.0, 50.0, 60.0, 60.0]
MISS = [100.0, 100.0, 110.0, 110.0]


def test_box_iou_known_values():
    a = torch.tensor([GT_A])
    b = torch.tensor([[0.0, 0.0, 10.0, 5.0], [5.0, 0.0, 15.0, 10.0], MISS])
    iou = box_iou_matrix(a, b)[0]
    assert iou.tolist() == pytest.approx([0.5, 1 / 3, 0.0], abs=1e-6)
    assert box_iou_matrix(a, torch.zeros(0, 4)).shape == (1, 0)


def test_ap_101_point_hand_computed():
    # Ranked hit, miss, hit against 2 ground truths: recall 0.5, 0.5, 1.0; precision 1, 0.5, 2/3.
    # 101-point AP = (51 levels at precision 1 + 50 levels at 2/3) / 101.
    s = _sample([GT_A, MISS, GT_B], [0.9, 0.8, 0.7], [0, 0, 0], [GT_A, GT_B], [0, 0])
    m = detection_metrics([s], num_classes=1)
    assert m["map50"] == pytest.approx((51 + 50 * 2 / 3) / 101)


def test_duplicate_detection_is_a_false_positive():
    s = _sample([GT_A, GT_A], [0.9, 0.8], [0, 0], [GT_A], [0])
    pr = precision_recall([s], iou_thr=0.5, score_thr=0.5)
    assert (pr["tp"], pr["fp"], pr["fn"]) == (1, 1, 0)
    assert pr["precision"] == 0.5 and pr["recall"] == 1.0
    # AP is unaffected: the duplicate ranks below the only true positive.
    assert detection_metrics([s], 1)["map50"] == pytest.approx(1.0)


def test_wrong_class_is_fp_and_fn():
    s = _sample([GT_A], [0.9], [1], [GT_A], [0])
    pr = precision_recall([s], iou_thr=0.5, score_thr=0.5)
    assert (pr["tp"], pr["fp"], pr["fn"]) == (0, 1, 1)


def test_score_threshold_drops_low_confidence():
    s = _sample([GT_A], [0.3], [0], [GT_A], [0])
    pr = precision_recall([s], iou_thr=0.5, score_thr=0.5)
    assert (pr["tp"], pr["fp"], pr["fn"]) == (0, 0, 1)
    assert pr["f1"] == 0.0


def test_classes_without_ground_truth_do_not_enter_the_mean():
    # Class 2 has a stray prediction and no ground truth; mAP averages classes 0 and 1 only.
    s = _sample([GT_A, MISS, MISS], [0.9, 0.9, 0.9], [0, 1, 2], [GT_A, GT_B], [0, 1])
    m = detection_metrics([s], num_classes=3)
    assert set(m["per_class_ap50"]) == {"0", "1"}
    assert m["map50"] == pytest.approx(0.5)


def test_map50_95_averages_over_iou_thresholds():
    # IoU 0.62 passes the 0.50, 0.55 and 0.60 thresholds and fails the other seven.
    s = _sample([[0.0, 0.0, 10.0, 6.2]], [0.9], [0], [GT_A], [0])
    m = detection_metrics([s], 1)
    assert m["map50"] == pytest.approx(1.0)
    assert m["map50_95"] == pytest.approx(0.3)


def test_matching_is_per_image():
    # The same box in another image must not match this image's ground truth.
    s1 = _sample([], [], [], [GT_A], [0], image_id="v:0")
    s2 = _sample([GT_A], [0.9], [0], [], [], image_id="v:1")
    m = detection_metrics([s1, s2], 1)
    assert m["map50"] == 0.0 and (m["tp"], m["fp"], m["fn"]) == (0, 1, 1)


def test_multilabel_report_macro_f1_and_exact_match():
    pred = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.1]])
    target = torch.tensor([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    rep = multilabel_report(pred, target, ["a", "b"])
    # a: F1 1.0. b: never present but predicted once, so it counts with F1 0.
    assert [c["name"] for c in rep["per_class"]] == ["a", "b"]
    assert rep["macro_f1"] == pytest.approx(0.5)
    assert rep["exact_match"] == pytest.approx(2 / 3)


def test_multilabel_report_skips_absent_and_unpredicted_class():
    pred = torch.tensor([[0.9, 0.1]])
    target = torch.tensor([[1.0, 0.0]])
    rep = multilabel_report(pred, target, ["a", "b"])
    assert [c["name"] for c in rep["per_class"]] == ["a"]
    assert rep["macro_f1"] == 1.0


def test_segment_iou_and_flicker():
    pred = torch.tensor([[1, 0], [1, 1], [0, 0]], dtype=torch.bool)
    target = torch.tensor([[1, 0], [0, 1], [0, 0]], dtype=torch.bool)
    assert segment_iou(pred, target) == pytest.approx(0.75)
    flags = torch.tensor([[1], [0], [0], [1]], dtype=torch.bool)
    assert flicker(flags) == pytest.approx(2 / 3)
    assert flicker(flags[:1]) == 0.0
