# Improvements log

## Round 1 (2026-11-05): tests for the metrics and temporal windows, CI, pytest from a clean clone

### What changed

1. **Tests for the code behind every README number.** Before this round the 8 tests covered splits, labels, a
   detector forward pass, one perfect and one missed box for mAP, the hold/track rule, and the arm. They added
   19 tests:
   - `tests/test_metrics.py` (11 tests), hand-computed:
     - 101-point AP for a hit/miss/hit ranking, (51 + 50·2/3)/101
     - mAP@0.5:0.95 for a box at IoU 0.62, which gives 3 of 10 thresholds, so 0.3
     - a duplicate detection counts as a false positive
     - a wrong class counts as one FP and one FN
     - the score threshold is applied
     - classes with no ground truth are left out of the mean
     - matching is per image
     - macro-F1 and exact match, including the rule that skips absent, unpredicted classes
     - segment IoU and flicker
   - `tests/test_temporal_windows.py` (3 tests): GRU windows never cross a video boundary, are built in frame
     order, and pad on the left. The windows built for training (`train_temporal.make_windows`) and for
     evaluation (`evaluate.presence_probs`) come from two separate copies of the same loop. A test now checks
     that they produce identical tensors.
   - `tests/test_detector_decode.py` (5 tests): box decoding from cell offsets, the single confident cell
     coming out of `predict`, boxes clamped to the image, NMS, and that NMS is per class.
2. **CI.** `.github/workflows/tests.yml` runs pyflakes and pytest on every push and pull request. It uses
   Python 3.10, the version the README names, and CPU-only PyTorch at the pinned versions.
3. **`python -m pytest` works from a clean clone.** `pyproject.toml` now sets `pythonpath = ["src"]`. Before,
   the suite errored at collection unless `PYTHONPATH=src` was exported.
4. **Lint.** Removed an unused import (`precision_recall`) and an unused variable (`key`) from
   `evaluate.py`. Neither changes behaviour.
5. **README.** A short "Tests" section that needs no data, GPU or ROS. The test command in "Reproduce" is now
   `.venv/bin/python -m pytest`.

### Why it matters

mAP, macro-F1 and the GRU windows are what every headline number comes from, and they had almost no tests. If
the training and evaluation windows drifted apart, the reported temporal results would quietly stop meaning
what they claim. There was no CI at all.

### How it was verified

Run in a fresh clone each time, with `PYTHONPATH` unset. Python 3.11 venv on Linux, CPU torch 2.14.0 from PyPI.

```
$ env -u PYTHONPATH python -m pytest          # main
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 1.51s
$ PYTHONPATH=src python -m pytest tests -q    # main, README's old command
8 passed

$ env -u PYTHONPATH python -m pytest          # this branch
27 passed in 3.32s

$ python -m pyflakes src tests ros2 | wc -l   # main -> branch
2 -> 0
```

PyPI lists `cp310` manylinux wheels for torch 2.14.0 and torchvision 0.29.0, so the Python 3.10 CI job can
install them. The PyTorch CPU index (`download.pytorch.org/whl/cpu`) that CI uses is blocked from this
sandbox, so the first CI run on this PR is the check for that step.

### Not verified

- None of the README's measured numbers were re-run here: the mAP values (0.321 / 0.286 / 0.241 / 0.304 /
  0.226), the macro-F1 values, 67.1 ms/frame, and the ROS replay counts. They were left unchanged. They need
  the SurgVU clips, and the latency is specific to an M1 with MPS.
- The ROS 2 demo: no ROS 2 Jazzy here.

### Files touched

`tests/test_metrics.py` (new), `tests/test_temporal_windows.py` (new), `tests/test_detector_decode.py` (new),
`.github/workflows/tests.yml` (new), `pyproject.toml`, `src/surgisense/evaluate.py`, `README.md`, `IMPROVEMENTS.md` (new)
