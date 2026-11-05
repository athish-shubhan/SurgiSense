"""Temporal windows: never cross a video boundary, and training and evaluation build the same ones."""

import torch

from surgisense.data import CLASSES
from surgisense.evaluate import presence_probs
from surgisense.train_temporal import make_windows


def _records():
    # Deliberately out of frame order, with two videos interleaved.
    order = [("a", 2), ("b", 0), ("a", 0), ("a", 1), ("b", 1)]
    return [{"video": v, "frame": f, "boxes": [[0, 0, 1, 1, i % len(CLASSES)]]} for i, (v, f) in enumerate(order)]


def _feats(records):
    # Feature = (video code, frame), so every window element can be traced back to its frame.
    return torch.tensor([[0.0 if r["video"] == "a" else 1.0, float(r["frame"])] for r in records])


def test_make_windows_stays_inside_one_video_and_pads_left():
    records = _records()
    x, y = make_windows(records, _feats(records), window=3)
    assert x.shape == (5, 3, 2) and y.shape == (5, len(CLASSES))
    rows = [w.tolist() for w in x]
    # Video a in frame order, then video b; short histories repeat the first frame on the left.
    assert rows[0] == [[0, 0], [0, 0], [0, 0]]
    assert rows[1] == [[0, 0], [0, 0], [0, 1]]
    assert rows[2] == [[0, 0], [0, 1], [0, 2]]
    assert rows[3] == [[1, 0], [1, 0], [1, 0]]
    assert rows[4] == [[1, 0], [1, 0], [1, 1]]
    for w in x:
        assert len(set(w[:, 0].tolist())) == 1  # one video per window


def test_make_windows_labels_follow_the_last_frame():
    records = _records()
    _, y = make_windows(records, _feats(records), window=3)
    # Window 0 ends at video a frame 0, which is records[2].
    assert y[0].argmax().item() == records[2]["boxes"][0][4]


class _Recorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, seq):
        self.seen.append(seq[0].clone())
        return torch.zeros(seq.shape[0], len(CLASSES))


def test_eval_windows_match_training_windows():
    records = _records()
    feats = _feats(records)
    train_x, _ = make_windows(records, feats, window=3)
    model = _Recorder()
    probs = presence_probs(model, {"kind": "temporal", "window": 3}, records, feats, "cpu")
    assert probs.shape == (5, len(CLASSES))
    assert torch.equal(torch.stack(model.seen), train_x)
