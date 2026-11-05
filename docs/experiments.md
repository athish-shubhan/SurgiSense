# Experiment records

Each training script writes `experiments/<id>/record.json` and `record.md`. The numbers below are copied from those records and from `evaluation/results.json`. Test metrics are not inside the training records, because the training scripts do not look at video 7 except to cache features.

Hardware for the later runs: Python 3.10.14, PyTorch 2.14.0, macOS arm64, MPS available. The baseline detector process reported MPS unavailable and trained on CPU.

## 20260930-024057-baseline

Hypothesis: a frame-wise stride-16 detector, without augmentation, localizes tools on a held-out clip.

Configuration: 8 epochs, batch 8, seed 7, no flip or color jitter, device CPU. Wall time 62.2 min.

Validation mAP@0.5 by epoch: 0.198, 0.226, 0.243, 0.226, 0.231, 0.282, 0.286, 0.286. Best is epoch 7, 0.286. Final training loss 0.280.

Test mAP@0.5, measured later by `evaluate` on MPS: 0.304. Test mAP@0.5:0.95: 0.083.

Checkpoint: `checkpoints/detector_baseline_best.pt`.

## 20260930-034326-aug

Hypothesis: horizontal flip and photometric jitter improve validation mAP@0.5.

Configuration: same schedule, augmentation on, device MPS. Wall time 38.9 min.

Validation mAP@0.5 by epoch: 0.242, 0.205, 0.283, 0.315, 0.305, 0.321, 0.301, 0.309. Best is epoch 6, 0.321. Final training loss 0.383.

Test mAP@0.5: 0.241. Test mAP@0.5:0.95: 0.061.

The hypothesis holds on the validation clip and fails on the test clip. The selected checkpoint is still this one, because selection is validation mAP@0.5. Checkpoint: `checkpoints/detector_aug_best.pt`.

## 20260930-042508-temporal

Hypothesis: a GRU over frozen frame features beats a linear frame classifier on tool presence.

Feature extraction: 96.7 s for train, validation, and test. Frame-head validation macro-F1: 0.491. GRU window 4: 0.422. GRU window 16: 0.493. Window 16 is `checkpoints/temporal_best.pt`.

Test macro-F1, from `evaluate`: frame head 0.234, window 16 0.294. Test predicted flicker: 0.053 and 0.019. Window 4 was not evaluated on the test clip.

## Independent CPU re-check (2026-11-05)

The two committed detector checkpoints were re-scored on a fresh download of the clips, on a different machine:
Linux x86_64, CPU only, Python 3.11, PyTorch 2.14.0 from PyPI. Preprocessing gave the same split sizes
(3795 / 512 / 871 frames).

```
python -m surgisense download && python -m surgisense preprocess
python -m surgisense evaluate --detectors-only        # 1 min 23 s wall on this CPU
```

| detector | val mAP@0.5 recorded (MPS) → CPU | test mAP@0.5 recorded (MPS) → CPU | test mAP@0.5:0.95 recorded → CPU |
| --- | --- | --- | --- |
| baseline | 0.286 → 0.290 | 0.304 → 0.300 | 0.083 → 0.083 |
| aug | 0.321 → 0.326 | 0.241 → 0.244 | 0.061 → 0.063 |

Every value is within 0.005 of the recorded one. The conclusions hold: augmentation wins on validation and
loses on the test clip, and vessel sealer AP@0.5 is 0 for both. The cause of the small gap (MPS vs CPU
kernels, or JPEG decoding on a different OpenCV build) was not isolated. The temporal and GRU numbers were not
re-checked, because the sandbox could not download the ImageNet MobileNetV3 weights that feature extraction
needs. The recorded figures above are unchanged.

## Evaluation and the arm

`evaluation/report.md` is the comparison table. `evaluation/results.json` is the same content with per-class average precision, the threshold sweep, and the confusion matrix.

Inference profile (`evaluation/inference/summary.json`): 67.1 ms/frame, 14.9 FPS, 863 timed frames, checkpoint `detector_aug_best.pt`, threshold 0.35, 454 hold frames, 417 track frames.

Simulation (`evaluation/sim_frames/trajectory.json`): MuJoCo, 871 steps.

ROS 2 replay (`evaluation/sim_frames/ros_trajectory.json`): MuJoCo, 871 frames, 417 track, 454 hold. Hold frames do not move the joints.
