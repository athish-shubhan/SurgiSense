"""Box decoding and NMS in the detector's post-processing."""

import torch

from surgisense.detector import _nms, decode_boxes, predict


def test_decode_boxes_cell_centre_and_size():
    raw = torch.zeros(1, 4, 4, 5)  # tx = ty = 0 -> centre of the cell; tw = th = 0 -> one stride wide
    boxes = decode_boxes(raw, stride=16)
    assert boxes.shape == (1, 4, 5, 4)
    assert boxes[0, 2, 3].tolist() == [48.0, 32.0, 64.0, 48.0]


def test_predict_returns_the_single_confident_cell():
    num_classes = 3
    pred = torch.zeros(1, 1 + num_classes + 4, 4, 5)
    pred[:, 0] = -20.0  # objectness off everywhere
    pred[0, 0, 2, 3] = 10.0  # except one cell
    pred[0, 1 + 1, 2, 3] = 10.0  # which is class 1
    dets = predict(pred, (64, 80))[0]
    assert dets["labels"].tolist() == [1]
    assert dets["boxes"][0].tolist() == [48.0, 32.0, 64.0, 48.0]
    assert 0.99 < float(dets["scores"][0]) <= 1.0


def test_predict_clamps_boxes_to_the_image():
    pred = torch.zeros(1, 1 + 1 + 4, 1, 1)
    pred[0, 0] = 10.0
    pred[0, -2:] = 4.0  # exp(4) * 16 px, much larger than the 16x16 image
    box = predict(pred, (16, 16))[0]["boxes"][0].tolist()
    assert box == [0.0, 0.0, 16.0, 16.0]


def test_nms_suppresses_overlaps_only():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 0.0, 11.0, 10.0], [50.0, 50.0, 60.0, 60.0]])
    scores = torch.tensor([0.8, 0.9, 0.7])
    assert _nms(boxes, scores, iou_thr=0.4).tolist() == [1, 2]
    assert _nms(torch.zeros(0, 4), torch.zeros(0), 0.4).numel() == 0


def test_predict_runs_nms_per_class():
    # Two identical boxes with different classes are both kept.
    pred = torch.zeros(1, 1 + 2 + 4, 1, 2)
    pred[0, 0] = 10.0
    pred[0, 1, 0, 0], pred[0, 2, 0, 1] = 10.0, 10.0
    pred[0, 3, 0, 1] = -100.0  # pull the second cell's centre left ...
    pred[0, 3, 0, 0] = 100.0   # ... and the first one's right, so the boxes overlap heavily
    dets = predict(pred, (16, 32))[0]
    assert sorted(dets["labels"].tolist()) == [0, 1]
