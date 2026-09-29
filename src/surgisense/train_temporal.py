"""Train a frame-wise presence baseline and GRU windows on frozen features."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from surgisense.data import CLASSES, FrameDataset, collate, load_config, load_index, records_for_split
from surgisense.experiment import device, write_experiment
from surgisense.metrics import multilabel_report
from surgisense.temporal import FeatureEncoder, FramePresence, TemporalPresence
from surgisense.train_detector import set_seed


def presence_of(records: list[dict]) -> torch.Tensor:
    y = torch.zeros(len(records), len(CLASSES))
    for i, rec in enumerate(records):
        for box in rec["boxes"]:
            y[i, int(box[4])] = 1.0
    return y


@torch.no_grad()
def extract(records: list[dict], dev, batch_size: int, image_size: tuple[int, int]) -> torch.Tensor:
    ds = FrameDataset(records, augment=False, image_size=image_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    encoder = FeatureEncoder(pretrained=True).to(dev)
    chunks = []
    for images, _ in loader:
        chunks.append(encoder(images.to(dev)).cpu())
    return torch.cat(chunks)


def make_windows(records: list[dict], feats: torch.Tensor, window: int) -> tuple[torch.Tensor, torch.Tensor]:
    by_video: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        by_video.setdefault(rec["video"], []).append(i)
    xs, ys = [], []
    labels = presence_of(records)
    for idxs in by_video.values():
        idxs = sorted(idxs, key=lambda i: records[i]["frame"])
        for t, i in enumerate(idxs):
            start = max(0, t - window + 1)
            seq_idx = idxs[start : t + 1]
            seq = feats[seq_idx]
            if len(seq) < window:
                pad = seq[:1].repeat(window - len(seq), 1)
                seq = torch.cat([pad, seq], 0)
            xs.append(seq)
            ys.append(labels[i])
    return torch.stack(xs), torch.stack(ys)


def train_head(model, x, y, epochs, lr, weight_decay, dev, batch_size) -> list[float]:
    pos = y.sum(0)
    weight = ((len(y) - pos) / pos.clamp(min=1)).clamp(max=20).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    losses = []
    model.train()
    for _ in range(epochs):
        total, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb, pos_weight=weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            n += 1
        losses.append(total / max(n, 1))
    return losses


@torch.no_grad()
def predict_split(model, x, dev, batch_size) -> torch.Tensor:
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        out.append(torch.sigmoid(model(x[i : i + batch_size].to(dev))).cpu())
    return torch.cat(out)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Train frame and temporal presence models")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--index", default="data/processed/index.json")
    args = p.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    index = load_index(args.index)
    dev = device()
    groups = {name: records_for_split(index, name) for name in ("train", "val", "test")}
    feats = {}
    t0 = time.perf_counter()
    for name, records in groups.items():
        print(f"features {name} {len(records)}")
        feats[name] = extract(records, dev, cfg["batch_size"], (cfg["image_height"], cfg["image_width"]))
    feature_seconds = time.perf_counter() - t0
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({k: v for k, v in feats.items()}, "data/processed/features.pt")

    tcfg = cfg["temporal"]
    y_train = presence_of(groups["train"])
    frame = FramePresence(feats["train"].shape[1], len(CLASSES)).to(dev)
    frame_losses = train_head(frame, feats["train"], y_train, tcfg["epochs"], tcfg["lr"], tcfg["weight_decay"], dev, tcfg["batch_size"])
    frame_val = predict_split(frame, feats["val"], dev, tcfg["batch_size"])
    frame_report = multilabel_report(frame_val, presence_of(groups["val"]), CLASSES)
    torch.save({"model": frame.state_dict(), "dim": feats["train"].shape[1], "kind": "frame"}, "checkpoints/temporal_frame.pt")

    results = {"frame_val_macro_f1": frame_report["macro_f1"], "windows": {}}
    best_f1, best_window = -1.0, None
    for window in tcfg["windows"]:
        xtr, ytr = make_windows(groups["train"], feats["train"], window)
        xva, yva = make_windows(groups["val"], feats["val"], window)
        model = TemporalPresence(xtr.shape[-1], tcfg["hidden"], len(CLASSES)).to(dev)
        losses = train_head(model, xtr, ytr, tcfg["epochs"], tcfg["lr"], tcfg["weight_decay"], dev, tcfg["batch_size"])
        pred = predict_split(model, xva, dev, tcfg["batch_size"])
        report = multilabel_report(pred, yva, CLASSES)
        path = f"checkpoints/temporal_w{window}.pt"
        torch.save({"model": model.state_dict(), "window": window, "hidden": tcfg["hidden"], "dim": xtr.shape[-1], "kind": "gru"}, path)
        results["windows"][str(window)] = {"val_macro_f1": report["macro_f1"], "exact_match": report["exact_match"], "final_loss": losses[-1]}
        print(f"window {window} val macro-F1 {report['macro_f1']:.3f}")
        if report["macro_f1"] > best_f1:
            best_f1, best_window = report["macro_f1"], window
            torch.save({"model": model.state_dict(), "window": window, "hidden": tcfg["hidden"], "dim": xtr.shape[-1], "kind": "gru"}, "checkpoints/temporal_best.pt")

    exp_id = time.strftime("%Y%m%d-%H%M%S-temporal")
    write_experiment(
        exp_id,
        {
            "hypothesis": "A GRU over frozen frame features beats a linear frame classifier on tool presence.",
            "method": f"Frozen MobileNetV3-Small pool, frame linear vs GRU windows {tcfg['windows']}",
            "configuration": tcfg,
            "dataset_version": "surgvu-public-tool-clips",
            "checkpoint": "checkpoints/temporal_best.pt",
            "result": {
                **results,
                "best_window": best_window,
                "feature_seconds": feature_seconds,
                "frame_losses_tail": frame_losses[-3:],
            },
            "interpretation": "The window with higher validation macro-F1 is temporal_best. Test metrics are in evaluate.",
        },
    )
    print(f"best window {best_window} val macro-F1 {best_f1:.3f}")


if __name__ == "__main__":
    main()
