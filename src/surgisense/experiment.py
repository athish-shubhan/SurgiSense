"""Small JSON experiment records. One directory per run."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


def hardware() -> dict:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "mps": bool(torch.backends.mps.is_available()),
        "cuda": bool(torch.cuda.is_available()),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def write_experiment(exp_id: str, payload: dict, root: str | Path = "experiments") -> Path:
    path = Path(root) / exp_id
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": exp_id,
        "created": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware(),
        **payload,
    }
    (path / "record.json").write_text(json.dumps(payload, indent=2))
    (path / "record.md").write_text(_markdown(payload))
    return path


def _markdown(payload: dict) -> str:
    lines = [
        f"# {payload['experiment_id']}",
        "",
        f"Hypothesis: {payload.get('hypothesis', '')}",
        "",
        f"Method: {payload.get('method', '')}",
        "",
        "## Result",
        "",
        "```json",
        json.dumps(payload.get("result", {}), indent=2),
        "```",
        "",
        f"Interpretation: {payload.get('interpretation', '')}",
        "",
    ]
    return "\n".join(lines)
