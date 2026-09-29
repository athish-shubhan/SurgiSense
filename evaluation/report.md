# SurgiSense evaluation

Test clip is video 7. Validation clip is video 4. Metrics below are computed from the trained checkpoints.

## Detection

| model | val mAP@0.5 | test mAP@0.5 | test mAP@0.5:0.95 | test F1 @ score 0.5 |
| --- | --- | --- | --- | --- |
| baseline | 0.286 | 0.304 | 0.083 | 0.227 |
| aug | 0.321 | 0.241 | 0.061 | 0.176 |

## Temporal presence

| model | val macro-F1 | test macro-F1 | test segment IoU | pred flicker | gt flicker |
| --- | --- | --- | --- | --- | --- |
| frame | 0.491 | 0.234 | 0.194 | 0.053 | 0.004 |
| best | 0.493 | 0.294 | 0.239 | 0.019 | 0.004 |

## Temporal gating of the selected detector

Selected detector: `aug`.

- ungated test mAP@0.5: 0.241
- score × temporal probability, test mAP@0.5: 0.226
- drop boxes whose temporal probability is below 0.5, test mAP@0.5: 0.191

## Uncertainty

Threshold selected on validation: 0.350.
A frame is uncertain when min(top-box score, temporal probability of that class) is below the threshold. The assistant holds in that state.

## Error analysis

```json
{
  "missed_boxes": 326,
  "border_misses": 16,
  "border_boxes": 101,
  "small_misses": 8,
  "small_boxes": 13,
  "motion_misses_diff_gt_15": 112,
  "forceps_family_confusions": 202,
  "rare_class_misses": 41,
  "median_blur_hit": 982.4618875992858,
  "median_blur_miss": 982.159586335975,
  "median_motion_hit": 12.833730697631836,
  "median_motion_miss": 13.124079704284668,
  "example_images": [
    "evaluation/failure_cases/forceps_confusion_7_0.jpg",
    "evaluation/failure_cases/motion_7_29.jpg",
    "evaluation/failure_cases/small_7_95.jpg",
    "evaluation/failure_cases/border_7_121.jpg",
    "evaluation/failure_cases/rare_7_546.jpg"
  ]
}
```

Green boxes in `evaluation/failure_cases/` are ground truth. Red boxes are predictions with score at least 0.2.

This is a research and simulation prototype. It is not clinically validated and it does not perform surgery.
