"""Stride-16 tool detector. One box per cell; the public clips do not put two tools in the same cell."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import batched_nms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class ToolDetector(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        # features[:6] is stride 16, 40 channels. At 256x320 that is a 16x20 grid.
        self.features = backbone.features[:6]
        self.num_classes = num_classes
        self.stride = 16
        self.head = nn.Sequential(
            nn.Conv2d(40, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1 + num_classes + 4, 1),
        )
        nn.init.normal_(self.head[-1].weight, std=0.01)
        bias = self.head[-1].bias
        nn.init.zeros_(bias)
        with torch.no_grad():
            bias[0] = -4.0  # objectness prior near zero
            bias[-2] = 0.8  # width prior ~ exp(0.8)*16 ≈ 36 px at the training resolution
            bias[-1] = 0.8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def decode_boxes(raw: torch.Tensor, stride: int) -> torch.Tensor:
    """raw is B,4,H,W (tx, ty, tw, th) -> B,H,W,4 xyxy."""
    _, _, gh, gw = raw.shape
    device = raw.device
    gy, gx = torch.meshgrid(
        torch.arange(gh, device=device),
        torch.arange(gw, device=device),
        indexing="ij",
    )
    tx, ty, tw, th = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
    cx = (gx + tx.sigmoid()) * stride
    cy = (gy + ty.sigmoid()) * stride
    w = tw.clamp(-2, 4).exp() * stride
    h = th.clamp(-2, 4).exp() * stride
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _ciou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    tl = torch.max(pred[:, :2], target[:, :2])
    br = torch.min(pred[:, 2:], target[:, 2:])
    wh = (br - tl).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area_p = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    area_t = (target[:, 2] - target[:, 0]).clamp(min=0) * (target[:, 3] - target[:, 1]).clamp(min=0)
    iou = inter / (area_p + area_t - inter + 1e-7)
    pc = (pred[:, :2] + pred[:, 2:]) / 2
    tc = (target[:, :2] + target[:, 2:]) / 2
    rho2 = ((pc - tc) ** 2).sum(dim=1)
    enc_tl = torch.min(pred[:, :2], target[:, :2])
    enc_br = torch.max(pred[:, 2:], target[:, 2:])
    c2 = ((enc_br - enc_tl) ** 2).sum(dim=1) + 1e-7
    pw = (pred[:, 2] - pred[:, 0]).clamp(min=1e-4)
    ph = (pred[:, 3] - pred[:, 1]).clamp(min=1e-4)
    tw = (target[:, 2] - target[:, 0]).clamp(min=1e-4)
    th = (target[:, 3] - target[:, 1]).clamp(min=1e-4)
    v = (4 / torch.pi**2) * (torch.atan(tw / th) - torch.atan(pw / ph)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)
    return 1 - iou + rho2 / c2 + alpha * v


def detection_loss(pred: torch.Tensor, batch: list[dict], stride: int = 16) -> dict[str, torch.Tensor]:
    b, _, gh, gw = pred.shape
    device = pred.device
    num_classes = pred.shape[1] - 5
    # Targets are built on the CPU and copied once. Indexing an MPS tensor box
    # by box forces a device sync per box.
    obj_t = torch.zeros(b, gh, gw)
    cls_t = torch.zeros(b, gh, gw, dtype=torch.long)
    box_t = torch.zeros(b, gh, gw, 4)
    for i, sample in enumerate(batch):
        boxes = sample["boxes"].cpu()
        if len(boxes) == 0:
            continue
        gx = ((boxes[:, 0] + boxes[:, 2]) * 0.5 // stride).long()
        gy = ((boxes[:, 1] + boxes[:, 3]) * 0.5 // stride).long()
        inside = (gx >= 0) & (gx < gw) & (gy >= 0) & (gy < gh)
        # In-order writes, so a later box in the same cell wins, as before.
        for k in inside.nonzero().flatten().tolist():
            obj_t[i, gy[k], gx[k]] = 1.0
            cls_t[i, gy[k], gx[k]] = sample["labels"][k]
            box_t[i, gy[k], gx[k]] = boxes[k]
    obj_t, cls_t, box_t = obj_t.to(device), cls_t.to(device), box_t.to(device)

    obj_logit = pred[:, 0]
    cls_logit = pred[:, 1 : 1 + num_classes]
    decoded = decode_boxes(pred[:, 1 + num_classes :], stride)

    ce = F.binary_cross_entropy_with_logits(obj_logit, obj_t, reduction="none")
    prob = obj_logit.sigmoid()
    p_t = prob * obj_t + (1 - prob) * (1 - obj_t)
    alpha = 0.25 * obj_t + 0.75 * (1 - obj_t)
    focal = alpha * (1 - p_t).pow(2) * ce
    npos = obj_t.sum().clamp(min=1)
    obj_loss = focal.sum() / npos

    pos = obj_t > 0.5
    if pos.any():
        cls_loss = F.cross_entropy(cls_logit.permute(0, 2, 3, 1)[pos], cls_t[pos])
        box_loss = _ciou(decoded[pos], box_t[pos]).mean()
    else:
        cls_loss = obj_logit.sum() * 0
        box_loss = obj_logit.sum() * 0
    total = obj_loss + cls_loss + box_loss
    return {"loss": total, "obj": obj_loss.detach(), "cls": cls_loss.detach(), "box": box_loss.detach()}


@torch.no_grad()
def predict(
    pred: torch.Tensor,
    image_hw: tuple[int, int],
    stride: int = 16,
    score_thresh: float = 0.01,
    nms_iou: float = 0.4,
    max_det: int = 40,
) -> list[dict]:
    height, width = image_hw
    num_classes = pred.shape[1] - 5
    obj = pred[:, 0].sigmoid()
    cls_prob = pred[:, 1 : 1 + num_classes].softmax(dim=1)
    boxes = decode_boxes(pred[:, 1 + num_classes :], stride)
    boxes[..., 0].clamp_(0, width)
    boxes[..., 2].clamp_(0, width)
    boxes[..., 1].clamp_(0, height)
    boxes[..., 3].clamp_(0, height)
    # NMS is a few dozen boxes per frame, so it runs on the CPU.
    obj, cls_prob, boxes = obj.cpu(), cls_prob.cpu(), boxes.cpu()
    out = []
    for i in range(pred.shape[0]):
        prob, lab = cls_prob[i].max(dim=0)
        score = obj[i] * prob
        mask = score > score_thresh
        if not mask.any():
            out.append(_empty())
            continue
        b, s, lb = boxes[i][mask], score[mask], lab[mask]
        keep = batched_nms(b, s, lb, nms_iou)[:max_det]
        out.append({"boxes": b[keep], "scores": s[keep], "labels": lb[keep]})
    return out


def _empty() -> dict:
    return {
        "boxes": torch.zeros(0, 4),
        "scores": torch.zeros(0),
        "labels": torch.zeros(0, dtype=torch.long),
    }
