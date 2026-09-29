# 20260930-042508-temporal

Hypothesis: A GRU over frozen frame features beats a linear frame classifier on tool presence.

Method: Frozen MobileNetV3-Small pool, frame linear vs GRU windows [4, 16]

## Result

```json
{
  "frame_val_macro_f1": 0.49090830019757903,
  "windows": {
    "4": {
      "val_macro_f1": 0.4217486404591335,
      "exact_match": 0.40234375,
      "final_loss": 0.010618550182941059
    },
    "16": {
      "val_macro_f1": 0.4930449555953575,
      "exact_match": 0.884765625,
      "final_loss": 0.012512418569531291
    }
  },
  "best_window": 16,
  "feature_seconds": 96.70036508300109,
  "frame_losses_tail": [
    0.07206102851778269,
    0.06964691380659739,
    0.06811827998608351
  ]
}
```

Interpretation: The window with higher validation macro-F1 is temporal_best. Test metrics are in evaluate.
