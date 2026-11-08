# SurgiSense study

Question: on a leakage-safe split of real endoscopic clips, can a small detector localize surgical tools, and does a short temporal model improve tool presence enough to change an uncertainty-gated assistant?

The answer from this machine is mixed. The detector learns the tools it sees. The temporal model improves presence on the held-out clip and reduces flicker. The augmentation that wins validation does not win the test clip. Gating detections with the temporal score does not raise detection mAP. The assistant still holds when confidence is low, and that hold is visible on the simulated arm.

## Data

Source: SurgVU public tool clips, direct download, seven videos at 1 fps, 640×512, COCO boxes. The 344 GB full archive was not downloaded. Case-level step and tool tables in the label zip use case ids that are not these seven clips, so they are not training targets.

5178 annotated frames, 11323 boxes, 0 corrupt frames. Eight tool names occur. Stapler, hook, tip-up, bipolar dissector, prograsp, and suction have no boxes in this release.

| split | videos | frames | boxes |
| --- | --- | --- | --- |
| train | 1, 2, 3, 5, 6 | 3795 | 8245 |
| validation | 4 | 512 | 1494 |
| test | 7 | 871 | 1584 |

Validation boxes are only cadiere forceps (512) and needle driver (982). Test boxes are bipolar forceps (867), monopolar curved scissor (563), clip applier (141), and vessel sealer (13). Vessel sealer is absent from train and validation. Clip applier has 117 training boxes. Grasping retractor and force bipolar appear in training and in neither validation nor test.

Video-level splitting prevents frame leakage. It also makes validation mAP a score on two tools that the test clip does not contain. That is a property of this seven-clip release, and the numbers below keep the two splits separate.

## Method

Detector: MobileNetV3-Small, cut at stride 16 (40 channels), then a 3×3 layer to 128 channels and a 1×1 head for objectness, class, and one box. Focal objectness, cross-entropy on occupied cells, CIoU boxes. ImageNet initialization. AdamW, trunk learning rate 1e-4, head 1e-3, cosine schedule, gradient clip 1, 8 epochs, batch 8, seed 7. Input is 320×256. Metrics are at that resolution. A Faster R-CNN MobileNet-V3-320-FPN step on this machine took about 4 to 7 seconds, which is why that model was not trained.

Ablation: the same schedule with and without horizontal flip and photometric jitter.

Temporal model: a frozen MobileNetV3-Small pool (576-d) and either a linear classifier or a GRU (hidden 128) over 4 or 16 frames. Binary cross-entropy with positive class weights. The GRU with the higher validation macro-F1 is kept. Feature extraction took 96.7 seconds.

Uncertainty: confidence is the minimum of the top detection score and the temporal probability of that class. The threshold is the lowest value on the validation clip with precision at least 0.75 and at least 30 frames. That sweep selected 0.35. Below it, the assistant holds. At or above it, the assistant commands a fixed posture for the dominant tool. Those postures are not surgical motions.

## Detection

| model | val mAP@0.5 | test mAP@0.5 | test mAP@0.5:0.95 | test F1 at score 0.5 |
| --- | --- | --- | --- | --- |
| no augmentation | 0.286 | 0.304 | 0.083 | 0.227 |
| flip and color jitter | 0.321 | 0.241 | 0.061 | 0.176 |

No-augmentation training ran on CPU because MPS was not available in that process (`experiments/20260930-024057-baseline`, 62.2 min, best validation mAP@0.5 0.286 at epoch 7). Augmented training ran on MPS (`experiments/20260930-034326-aug`, 38.9 min, best validation mAP@0.5 0.321 at epoch 6). Evaluation reloads both checkpoints on MPS.

Test AP@0.5 by class, classes with at least one test box:

| class | boxes | no augmentation | augmented |
| --- | --- | --- | --- |
| bipolar forceps | 867 | 0.564 | 0.570 |
| monopolar curved scissor | 563 | 0.581 | 0.374 |
| clip applier | 141 | 0.072 | 0.022 |
| vessel sealer | 13 | 0.000 | 0.000 |

Augmentation raised validation mAP, which is the selection rule, and lowered test mAP. The drop is concentrated on the monopolar curved scissor and the clip applier. Vessel sealer is undetected by both models. Bipolar forceps is essentially unchanged.

## Presence

| model | val macro-F1 | test macro-F1 | test segment IoU | predicted flicker | ground-truth flicker |
| --- | --- | --- | --- | --- | --- |
| linear frame head | 0.491 | 0.234 | 0.194 | 0.053 | 0.004 |
| GRU, window 4 | 0.422 |  |  |  |  |
| GRU, window 16 | 0.493 | 0.294 | 0.239 | 0.019 | 0.004 |

Window 4 is worse than the frame head on validation, so it is not selected. Window 16 is the selected temporal model. Its validation margin over the frame head is 0.002 macro-F1. On the test clip the margin is larger (0.294 vs 0.234), and predicted flicker falls from 0.053 to 0.019. Ground-truth flicker on that clip is 0.004, so the GRU is steadier than the frame head and still flickers more than the labels.

Using the GRU to rescale or drop detector boxes does not help localization of the selected detector:

- ungated test mAP@0.5: 0.241
- score multiplied by temporal probability: 0.226
- drop if temporal probability < 0.5: 0.191

## Uncertainty and the arm

On the test clip, with the augmented detector and the 16-frame GRU, the dataset loader counts 461 hold frames and 410 track frames. Sequential inference on `7_fps1.mp4` counts 454 hold and 417 track. ROS 2 replay of that inference log matches the video count: 417 track, 454 hold, backend `mujoco`, 871 messages received. Consecutive hold frames keep the same joint position (0 changes in the ROS trajectory).

The assistant node now steps the arm on its own 50 Hz timer and holds when no new context arrived for 0.25 s (ADR 0001). Recorded runs of that version are in `evaluation/ros/` (`nominal_summary.json`, `stall_summary.json`). The replay counts above come from the earlier message-driven node and `evaluation/sim_frames/ros_trajectory.json`; `tests/test_recorded_replay.py` reproduces them offline.

The operating threshold 0.35 comes from the validation sweep in `evaluation/threshold.json`.

## Errors

On the test clip, at IoU 0.5 and score 0.05, 326 ground-truth boxes are missed. Of those, 202 are confusions inside the forceps family, 112 coincide with frame-to-frame intensity change above 15, 16 touch the image border (101 border boxes exist), 8 are smaller than 32 px on the original frame (13 such boxes exist), and 41 are rare-class misses (clip applier, vessel sealer, grasping retractor). Median Laplacian variance is 982 on hits and 982 on misses, so blur does not separate them. Median motion is 12.8 on hits and 13.1 on misses.

Example overlays, green ground truth and red predictions, are written to `evaluation/failure_cases/` by `evaluate`. They contain SurgVU frames, so that folder is gitignored and not published.

## Speed

Sequential pass over video 7 on MPS, batch 1, resolution of the file 512×640, network input 320×256. Eight warmup frames are excluded. 863 frames remain.

| stage | ms |
| --- | --- |
| preprocess | 2.8 |
| detector | 9.7 |
| temporal encoder and GRU | 22.3 |
| decode and NMS | 32.2 |
| total | 67.1 (14.9 FPS) |

Peak RSS was 477 MB. MPS allocation after the pass was 6.7 MB. These are one measured pass, not a multi-run benchmark.

## Limits

Seven clips. Validation and test contain different tools, and the checkpoint was chosen on validation even though it is worse on the test clip. Vessel sealer has 13 boxes, all in the test clip. The step labels in the label zip do not line up with these files, so phase recognition was not trained. The arm only switches between fixed postures in simulation.
