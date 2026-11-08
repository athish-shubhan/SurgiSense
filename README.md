# SurgiSense

Research and simulation only: this is not a clinical device and has not been evaluated on patients or in an operating room.

SurgiSense finds surgical tools in 1 fps endoscopic clips, estimates which tools are present over a short window, and publishes that state on ROS 2. If the confidence is below a threshold chosen on the validation clip, a two-link MuJoCo arm holds. Otherwise it moves to a fixed posture for the top tool. The postures are labels for the simulator, not surgical motions.

Measured on the public SurgVU tool clips, Apple M1, PyTorch 2.14. The augmented detector was selected because its validation mAP@0.5 was 0.321, against 0.286 with no augmentation. On the held-out clip that order reverses: 0.241 versus 0.304. A 16-frame GRU beat a linear presence head on the test clip (macro-F1 0.294 versus 0.234) by a small margin on validation (0.493 versus 0.491). Using the GRU to rescale detections lowered test mAP@0.5 from 0.241 to 0.226. Details and per-class numbers are in `docs/research.md`.

Sequential inference on the test clip, batch 1, MPS, 863 frames after a warmup of 8: 67.1 ms/frame (14.9 FPS). The ROS replay of that log stepped the MuJoCo arm for all 871 frames: 417 track, 454 hold. Hold frames do not move the joints.

## Data

[SurgVU](https://storage.googleapis.com/isi-surgvu/cat1_test_set_public.zip) publishes seven tool clips (about 164 MB) and a separate label zip. Both download with no registration. The full training archive is about 344 GB and is not used. The case-level `tasks.csv` / `tools.csv` files in the label zip do not match these seven clip ids, so phase labels are not trained. Presence labels come from the boxes.

The clips are someone else's dataset, released under the SurgVU terms, which apply to any use of the data and to models trained on it (check the dataset's download page for the licence before reusing it). Cite the dataset as: A. Zia, M. Berniker, R. Nespolo, X. Zhang, C. Perreault et al., "Surgical Visual Understanding (SurgVU) Dataset", arXiv:2501.09209, 2025 (Intuitive Surgical). This repo does not redistribute the video, the extracted frames, or overlays drawn on those frames. `download` fetches them into `data/raw/`, which is gitignored.

Split by video, so no frame is in two splits:

| split | videos | frames | tools in the boxes |
| --- | --- | --- | --- |
| train | 1, 2, 3, 5, 6 | 3795 | six tools; no vessel sealer |
| val | 4 | 512 | cadiere forceps, needle driver |
| test | 7 | 871 | bipolar forceps, monopolar curved scissor, clip applier, vessel sealer |

Stored frames are 640×512. The network sees 320×256. Metrics are computed at 320×256. Drawings scale the boxes back.

Validation and test do not share tools, so the two mAP numbers are not the same measurement. Selection still uses validation only.

## Models

Detector: MobileNetV3-Small cut at stride 16, then a 3×3 layer (128 channels) and a 1×1 head for objectness, class, and one box. Focal loss on objectness, cross-entropy on occupied cells, CIoU on boxes. ImageNet init, AdamW, 8 epochs, batch 8, seed 7. One run without augmentation, one with horizontal flip and color jitter.

Temporal: frozen MobileNetV3-Small pooled features. A linear head, and a GRU (hidden 128) at window 4 and window 16. The higher validation macro-F1 is kept. Window 16 won. Window 4 was worse than the linear head on validation (0.422) and was not run on the test clip.

Confidence for the arm is `min(top box score, temporal probability of that class)`. The validation sweep in `evaluation/threshold.json` picked 0.35.

## Tests

The tests use synthetic tensors and need no SurgVU download, no GPU and no ROS:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

Install the dev requirements to run the whole suite (`pip install -r requirements-dev.txt`, which adds `pycocotools`); `tests/test_metrics_reference.py` compares the mAP code with the COCO evaluator on randomized scenes and is skipped if `pycocotools` is missing. They cover the mAP / precision-recall / macro-F1 code against hand-computed cases, box decoding and per-class NMS, the per-video temporal windows (and that training and evaluation build identical windows), the hold/track rule, and the MuJoCo arm. `tests/test_recorded_replay.py` replays the committed `evaluation/inference/predictions.jsonl` through the same context, assistant and arm code the ROS nodes use, and checks the 417 track / 454 hold counts and the logged arm trajectory without a model or ROS. CI (`.github/workflows/tests.yml`) runs pyflakes and the suite on every push and pull request, on Linux with CPU-only PyTorch.

## Reproduce

Python 3.10. ROS 2 Jazzy is only needed for the topic demo; it lives in `.rosenv` and is not part of training.

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export PYTHONPATH=src MPLCONFIGDIR=$PWD/.mpl PYTORCH_ENABLE_MPS_FALLBACK=1
.venv/bin/python -m surgisense download
.venv/bin/python -m surgisense preprocess
.venv/bin/python -m surgisense explore
.venv/bin/python -m surgisense train-detector --no-augment --tag baseline
.venv/bin/python -m surgisense train-detector --tag aug
.venv/bin/python -m surgisense train-temporal
.venv/bin/python -m surgisense evaluate
.venv/bin/python -m surgisense infer
.venv/bin/python -m surgisense simulate
.venv/bin/python -m surgisense dashboard --serve
bash ros2/run_demo.sh 20
.venv/bin/python -m pytest
```

Checkpoints from the recorded runs are in `checkpoints/`, so evaluation and inference can be repeated without training if the clips are downloaded. `evaluate` extracts the frozen features itself when `data/processed/features.pt` is missing; it no longer requires `train-temporal`, which would retrain and overwrite the temporal checkpoints. `evaluate --detectors-only` re-scores just the two detectors, prints the metrics as JSON on stdout (progress goes to stderr, so it pipes into `jq`), writes nothing and needs no ImageNet weights. A CPU re-check of the detector numbers on another machine is in `docs/experiments.md`. The baseline run in `experiments/20260930-024057-baseline` used CPU because that process did not see MPS. Later runs used MPS.

## ROS 2 and the arm

`ros2/run_demo.sh` starts three processes. They only share topics (`std_msgs/String`, best effort):

- `/surgical/perception` replays `evaluation/inference/predictions.jsonl`
- `/surgical/context` rebuilds the state at the saved threshold
- `/surgical/assistant` steps the arm on a 50 Hz timer, holds when no context arrived for 0.25 s, and writes `evaluation/ros/assistant_log.json` and `assistant_log_summary.json`. The committed `evaluation/sim_frames/ros_trajectory.json` and the 417 / 454 counts come from the earlier message-driven version of this node. `bash ros2/run_demo.sh 15 --stall-after 300 --stall-seconds 2` injects a perception stall; recorded runs of the timer-driven node are in `evaluation/ros/*_summary.json` (ADR 0001)

On this Mac, Fast DDS stayed at zero subscribers. The script uses CycloneDDS on `lo0` with shared memory off (`ros2/cyclonedds.xml`). The logged backend for the full replay was `mujoco` (MuJoCo 3.3.7). If `mujoco` does not import, the same hold/track rule runs on a kinematic step and the log says `kinematic`.

## Limits

Seven clips. Vessel sealer has 13 boxes, all in the test clip, and both detectors score AP@0.5 of 0 on it. Augmentation helped validation and hurt the test clip. Temporal gating did not help detection mAP. Nothing here is a clinical device, and the arm does not operate on tissue.

`docs/architecture.md` is the layout. `docs/adr/` has two longer decision records (the control loop and watchdog; faster post-processing and a pycocotools-checked mAP). `docs/decisions.md` is why the detector, the GRU, and CycloneDDS were chosen. `docs/experiments.md` points at the run records.
