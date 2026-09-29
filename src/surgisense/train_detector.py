"""Train the frame-wise tool detector."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from surgisense.data import CLASSES, FrameDataset, collate, load_config, load_index, records_for_split
from surgisense.detector import ToolDetector, detection_loss, predict
from surgisense.experiment import device, write_experiment
from surgisense.metrics import detection_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, dev, opt, stride: int) -> dict:
    model.train()
    totals = {"loss": 0.0, "obj": 0.0, "cls": 0.0, "box": 0.0}
    n = 0
    for images, batch in loader:
        images = images.to(dev)
        pred = model(images)
        losses = detection_loss(pred, batch, stride)
        opt.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        for k in totals:
            totals[k] += float(losses[k].detach())
        n += 1
        if n % 100 == 0:
            print(f"  step {n} loss {totals['loss'] / n:.3f}", flush=True)
    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate_loader(model, loader, dev, cfg) -> dict:
    model.eval()
    samples = []
    for images, batch in loader:
        pred = model(images.to(dev))
        dets = predict(
            pred,
            (cfg["image_height"], cfg["image_width"]),
            stride=cfg["stride"],
            score_thresh=cfg["score_thresh"],
            nms_iou=cfg["nms_iou"],
            max_det=cfg["max_det"],
        )
        for det, sample in zip(dets, batch):
            samples.append(
                {
                    "image_id": f"{sample['video']}:{sample['frame']}",
                    "pred_boxes": det["boxes"],
                    "pred_scores": det["scores"],
                    "pred_labels": det["labels"],
                    "gt_boxes": sample["boxes"],
                    "gt_labels": sample["labels"],
                }
            )
    metrics = detection_metrics(samples, len(CLASSES))
    return metrics


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Train the SurgiSense detector")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--index", default="data/processed/index.json")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--tag", default="aug")
    p.add_argument("--resume")
    p.add_argument("--from-scratch", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    epochs = args.epochs or cfg["epochs"]
    batch_size = args.batch_size or cfg["batch_size"]
    augment = not args.no_augment
    set_seed(cfg["seed"])
    index = load_index(args.index)
    size = (cfg["image_height"], cfg["image_width"])
    train_ds = FrameDataset(records_for_split(index, "train"), augment=augment, image_size=size)
    val_ds = FrameDataset(records_for_split(index, "val"), augment=False, image_size=size)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("empty train or val split; run preprocess first")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=cfg["workers"], collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=cfg["workers"], collate_fn=collate)
    dev = device()
    model = ToolDetector(len(CLASSES), pretrained=not args.from_scratch).to(dev)
    backbone_ids = set(id(p) for p in model.features.parameters())
    groups = [
        {"params": [p for p in model.parameters() if id(p) in backbone_ids], "lr": cfg["lr_backbone"]},
        {"params": [p for p in model.parameters() if id(p) not in backbone_ids], "lr": cfg["lr_head"]},
    ]
    opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    start = 1
    best = -1.0
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start = int(ckpt.get("epoch", 0)) + 1
        best = float(ckpt.get("best_map50", -1))
        print(f"resumed at epoch {start}")

    exp_id = time.strftime(f"%Y%m%d-%H%M%S-{args.tag}")
    history = []
    t0 = time.perf_counter()
    for epoch in range(start, epochs + 1):
        random.seed(cfg["seed"] + epoch)
        t_ep = time.perf_counter()
        train_loss = run_epoch(model, train_loader, dev, opt, cfg["stride"])
        val = evaluate_loader(model, val_loader, dev, cfg)
        sched.step()
        row = {"epoch": epoch, "train": train_loss, "val_map50": val["map50"], "val_map50_95": val["map50_95"], "seconds": time.perf_counter() - t_ep}
        history.append(row)
        print(
            f"epoch {epoch} loss {train_loss['loss']:.3f} "
            f"map50 {val['map50']:.3f} map50_95 {val['map50_95']:.3f} "
            f"({row['seconds']:.0f}s)"
        )
        if val["map50"] > best:
            best = val["map50"]
        last = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": epoch,
            "best_map50": best,
            "classes": CLASSES,
            "augment": augment,
            "config": cfg,
            "tag": args.tag,
        }
        torch.save(last, ckpt_dir / f"detector_{args.tag}_last.pt")
        if val["map50"] >= best:
            torch.save(
                {k: v for k, v in last.items() if k != "optimizer"},
                ckpt_dir / f"detector_{args.tag}_best.pt",
            )
    elapsed = time.perf_counter() - t0
    result = {"history": history, "best_val_map50": best, "seconds": elapsed, "augment": augment, "device": str(dev)}
    write_experiment(
        exp_id,
        {
            "hypothesis": (
                "Horizontal flip and photometric jitter improve validation mAP@0.5."
                if augment
                else "A frame-wise stride-16 detector, without augmentation, localizes tools on a held-out clip."
            ),
            "method": f"MobileNetV3-Small stride-16 detector, tag={args.tag}, augment={augment}, epochs={epochs}, batch={batch_size}",
            "configuration": {"tag": args.tag, "augment": augment, "epochs": epochs, "batch_size": batch_size, "seed": cfg["seed"]},
            "dataset_version": "surgvu-public-tool-clips",
            "checkpoint": str(ckpt_dir / f"detector_{args.tag}_best.pt"),
            "result": result,
            "interpretation": "Validation mAP is the selection metric. Test numbers are produced by evaluate.",
        },
    )
    (Path("experiments") / exp_id / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best val mAP@0.5 {best:.4f} in {elapsed/60:.1f} min -> experiments/{exp_id}")


if __name__ == "__main__":
    main()
