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

## Round 2 (2026-11-05): evaluate the committed checkpoints without retraining, independent CPU re-check

Stacked on round 1 (`improve/round-1`).

### What changed

1. **`evaluate` no longer needs `train-temporal`.** It used to exit with "missing data/processed/features.pt.
   Run train_temporal first". `train_temporal` retrains the frame and GRU heads and overwrites
   `checkpoints/temporal_*.pt`, so the committed checkpoints the README offers could not be evaluated as they
   were. `evaluate` now extracts the frozen-encoder features for val and test itself (no training) and caches
   them (`load_or_extract_features`).
2. **`python -m surgisense evaluate --detectors-only`.** It re-scores the two committed detector checkpoints on
   val and test, prints mAP@0.5 and mAP@0.5:0.95 as JSON, writes nothing, and needs no ImageNet download. The
   detector loop was moved into `score_detectors()` so both paths share it.
3. **The headline detector numbers were re-measured** on a different machine (CPU, Linux) and recorded in
   `docs/experiments.md` next to the original MPS figures. The README's numbers are unchanged.
4. **Tests.** `tests/test_evaluate.py` (3 tests, synthetic JPEG frames and random-weight checkpoints):
   - `score_detectors` end to end
   - `evaluate --detectors-only` prints metrics and leaves the working tree byte-for-byte unchanged
   - features are extracted once, then reused from the cache

### Why it matters

A reader who downloads the clips should be able to check the README's numbers from the committed checkpoints.
Before this change, the only route was to retrain the temporal models first. Now it is one command, and the
detector numbers have been reproduced independently.

### How it was verified

```
$ python -m pytest                                   # round 1 -> round 2
27 passed  ->  30 passed in 3.31s
$ python -m pyflakes src tests ros2                  # clean

$ python -m surgisense download && python -m surgisense preprocess
{"train": 3795, "val": 512, "test": 871}             # same split sizes as the README table

$ time python -m surgisense evaluate --detectors-only
baseline val mAP@0.5 0.290, test mAP@0.5 0.300
aug val mAP@0.5 0.326, test mAP@0.5 0.244
real 1m22.655s
```

| detector | val mAP@0.5, recorded (MPS) → CPU | test mAP@0.5, recorded → CPU | test mAP@0.5:0.95, recorded → CPU |
| --- | --- | --- | --- |
| baseline | 0.286 → 0.290 | 0.304 → 0.300 | 0.083 → 0.083 |
| aug | 0.321 → 0.326 | 0.241 → 0.244 | 0.061 → 0.063 |

Every value is within 0.005 of the recorded one. The README's conclusions hold: augmentation wins on validation
and loses on test, and vessel sealer AP is 0 for both.

The full `python -m surgisense evaluate` was also run. It now reaches feature extraction instead of exiting,
then stops on `URLError: Tunnel connection failed: 403` for `download.pytorch.org`. That host is blocked in this
sandbox. Nothing was written before the failure.

### Not verified

- The temporal numbers: frame head macro-F1 0.234, GRU 0.294, validation 0.491 / 0.493 / 0.422, and the gated
  mAP of 0.226. The threshold 0.35 and the hold/track counts were not re-run either. All of these need the
  ImageNet MobileNetV3 weights, which the sandbox cannot download.
- The cause of the ≤0.005 mAP gap between the recorded MPS run and this CPU run was not isolated. It could be
  the kernels, or JPEG decoding with a different OpenCV build.
- 67.1 ms/frame on an M1 with MPS: there is no Apple hardware here.

### Files touched

`src/surgisense/evaluate.py`, `tests/test_evaluate.py` (new), `docs/experiments.md`, `README.md`, `IMPROVEMENTS.md`

## Round 3 (2026-11-05): the ROS replay numbers checked in CI; node logic testable without ROS

Stacked on round 2 (`improve/round-2`).

### What changed

1. **`ContextTracker` (`surgisense/context.py`).** The context node kept its history and previous-uncertainty
   flag in `nonlocal` variables inside a ROS callback, so that logic could only run under `rclpy`. It is now a
   small class with `update(message) -> SurgicalState`. `ros2/context_node.py` calls it, with no change in
   behaviour.
2. **The assistant node's special case is gone.** `ros2/assistant_node.py` built its own hold command when
   the state was uncertain or had no tools, and otherwise called `assistant_command(state, 0.0)`. That function
   already returns the identical hold command in those two cases, because confidence is never below 0. The node
   now calls it directly. The replay test below confirms the commands are identical on all 871 recorded frames.
3. **`tests/test_recorded_replay.py` (4 tests).** It replays the committed `evaluation/inference/predictions.jsonl`
   (what the perception node publishes) through `ContextTracker`, `assistant_command` and the MuJoCo `ArmSim`,
   with no model and no ROS, and checks:
   - 871 frames, **417 track / 454 hold**, the README's numbers
   - every command equals the one `infer` logged
   - modes and joint angles match the assistant node's logged `ros_trajectory.json` to `atol=1e-9`
   - hold frames never move the arm

### Why it matters

The README's ROS claims ("stepped the MuJoCo arm for all 871 frames: 417 track, 454 hold. Hold frames do not
move the joints") were backed only by a log file from one run on one Mac. Now CI re-derives them from the
committed artifacts on every push, so a change to the context or arm code that alters them fails a test.

### How it was verified

```
$ python -m pytest                                  # round 2 -> round 3
30 passed  ->  34 passed in 4.39s
$ python -m pyflakes src tests ros2                 # clean
```

Before changing any code, the same replay was run as a standalone script on round 2's code. That shows the
claims held already, and that the refactor did not change them:

```
frames 871 871 hold 454 track 417
mode mismatches vs ROS log 0 vs infer log 0 max joint diff 1.3877787807814457e-16
```

The recorded trajectory came from macOS arm64. This replay ran on Linux x86_64 with the same pinned MuJoCo 3.3.7,
and the joint angles agree to 1.4e-16.

### Not verified

- The ROS nodes themselves were not launched: there is no ROS 2 Jazzy here. `context_node.py` and
  `assistant_node.py` now delegate to the tested functions, but the `rclpy` wiring around them is untested.
- `infer` was not re-run, because it needs the ImageNet MobileNetV3 weights, which the sandbox cannot
  download. The replay starts from its committed output.

### Files touched

`src/surgisense/context.py`, `ros2/context_node.py`, `ros2/assistant_node.py`, `tests/test_recorded_replay.py` (new),
`README.md`, `IMPROVEMENTS.md`

## Round 4 (2026-11-05): review of rounds 1–3, and two fixes it found

Rounds 1–3 (https://github.com/athish-shubhan/SurgiSense/pull/1, https://github.com/athish-shubhan/SurgiSense/pull/2,
https://github.com/athish-shubhan/SurgiSense/pull/3) are merged. This round is a second pass over their merged
diff, and its PR is a fresh branch from `main`.

### What changed

1. **`evaluate --detectors-only` now prints valid JSON.** Round 2 said the command "prints … as JSON", but
   the two progress lines (`baseline val mAP@0.5 …`) went to stdout first, so `… | jq` failed. Round 2's test
   only passed because it sliced the output from the first `{`. Progress now goes to stderr, and the test
   parses stdout as-is.
2. **One temporal-window helper instead of three copies.** Round 1 tested that training (`make_windows`) and
   evaluation (`presence_probs`) build identical GRU windows, but `infer.py` had a third copy of the logic
   that nothing tested. `temporal.left_padded_window` is now the only implementation, and all three call it.
3. **Inference history is bounded.** `infer.py` appended every frame's feature vector to a list and only ever
   read the last `window` entries. It now uses `deque(maxlen=window)`, so memory no longer grows with the
   length of the stream.

### Why it matters

A README command that does not do what it says is worse than no command. The GRU's results depend on the
windows being built the same way in training and at inference, and only two of the three paths were checked.

### How it was verified

```
$ python -m pyflakes src tests ros2        # clean
$ python -m pytest
36 passed in 4.53s                         # was 34
```

New and changed tests:

- `test_left_padded_window_short_exact_and_long_histories`
- `test_streaming_inference_windows_match_training_windows`: the bounded-deque streaming path used by
  `infer.py` produces exactly the tensors `make_windows` produces.
- `test_detectors_only_prints_metrics_and_writes_nothing`, which now runs `json.loads(stdout)` with nothing
  sliced off. With the fix stashed it fails with
  `JSONDecodeError: Expecting value: line 1 column 1 (char 0)`.

On the real SurgVU clips (CPU):

```
$ python -m surgisense evaluate --detectors-only 2>err.txt | python -c "import json,sys; ..."
{'baseline': {'val': 0.2897, 'test': 0.3003}, 'aug': {'val': 0.3263, 'test': 0.2438}}
```

These are identical to round 2's numbers, so the refactor changed no detector result. The window refactor is
covered by the existing train/eval equality test plus the new streaming test.

### Not verified

- `infer` itself was not run: it needs the ImageNet MobileNetV3 weights, which the sandbox cannot download.
  Its new code path is covered only by the streaming-equality test, which uses the same helper and deque.
- Retraining the temporal heads was not repeated after the `make_windows` refactor. The unchanged equality
  tests show the windows are byte-for-byte identical, so training would see the same inputs.

### Files touched

`src/surgisense/temporal.py`, `src/surgisense/train_temporal.py`, `src/surgisense/evaluate.py`,
`src/surgisense/infer.py`, `tests/test_temporal_windows.py`, `tests/test_evaluate.py`, `README.md`, `IMPROVEMENTS.md`
