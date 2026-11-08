# ADR 0002: Fast post-processing and a reference-checked mAP

Status: accepted

## Context

Per-epoch validation and `evaluate` were slow, and it was not obvious why. On the augmented checkpoint (before this change):

| step | validation clip, 512 frames | test clip, 871 frames |
| --- | --- | --- |
| forward pass plus NMS | 107.1 s | 254.9 s |
| mAP@0.5 and mAP@0.5:0.95 | 7.8 s | 26.3 s |

The forward pass is cheap. The cost was NMS written as a Python loop over MPS tensors. Each IoU call and each `int(order[0])` forced a GPU sync. The training loss had the same pattern: targets were assigned with `.item()` on MPS tensors, one sync per box.

mAP was hand-written. Nothing checked it against a reference.

## Options

1. Use `pycocotools` or `torchmetrics` for mAP at run time.
2. Keep the small implementation, make it fast, and check it against pycocotools in tests.

For NMS, `torchvision.ops.batched_nms` is already a dependency through torchvision.

## Decision

- NMS: copy the decoded grid (16×20 cells) to the CPU once per batch and call `batched_nms`. Suppression is still per class at IoU > 0.4.
- Loss targets: built on the CPU, copied to the device once per batch. When two boxes land in one cell, the later one still wins, as before.
- mAP: the IoU matrix is computed once per image and class. Greedy matching runs per image for all ten IoU thresholds. AP uses a cumulative precision envelope and `searchsorted` over the 101 recall levels. `pycocotools` is a dev dependency only. `tests/test_metrics_reference.py` compares the two on randomized scenes with duplicates, misses, and wrong classes.

## Result

Measured on the same checkpoint and frames:

| step | before | after |
| --- | --- | --- |
| validation forward plus NMS | 107.1 s | 7.4 s |
| test forward plus NMS | 254.9 s | 19.2 s |
| validation metrics | 7.8 s (10.9 s on a rerun) | 0.69 s |
| test metrics | 26.3 s (21.4 s on a rerun) | 1.55 s |

The numbers did not move. On real predictions, old code, new code, and pycocotools agree to six decimals: validation mAP@0.5 0.320620 and mAP@0.5:0.95 0.113721; test 0.241209 and 0.061381. Old and new differ by about 4e-16, which is summation order.

## Consequences

- The run-to-run noise in the "before" timings is large because this 8 GB machine was swapping. The ratio is far bigger than that noise.
- pycocotools applies a 100-detection cap per image and class. The test scenes stay under it, and `predict` caps at 40 per image, so the cap never applies.
