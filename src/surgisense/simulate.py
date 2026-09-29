"""Replay an inference jsonl through the assistant arm and save the trajectory."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from surgisense.sim import ArmSim


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Step the assistant arm from inference output")
    p.add_argument("--predictions", default="evaluation/inference/predictions.jsonl")
    p.add_argument("--out", default="evaluation/sim_frames")
    p.add_argument("--backend", default="auto")
    args = p.parse_args()
    path = Path(args.predictions)
    if not path.exists():
        raise SystemExit(f"missing {path}. Run inference first.")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    sim = ArmSim(backend=args.backend)
    traj = []
    for row in rows:
        cmd = row["command"]
        q = sim.step(cmd["joints"])
        traj.append({"frame": row["state"]["frame"], "mode": cmd["mode"], "q": [float(v) for v in q], "tool": cmd["tool"]})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"backend": sim.backend, "steps": len(traj), "trajectory": traj}
    (out / "trajectory.json").write_text(json.dumps(payload))
    qs = np.asarray([t["q"] for t in traj])
    modes = [t["mode"] for t in traj]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(qs[:, 0], label="joint 1")
    ax.plot(qs[:, 1], label="joint 2")
    holds = [i for i, m in enumerate(modes) if m == "hold"]
    if holds:
        ax.scatter(holds[:: max(1, len(holds) // 40)], qs[holds[:: max(1, len(holds) // 40)], 0], s=8, c="0.4", label="hold")
    ax.set_xlabel("frame")
    ax.set_ylabel("joint position")
    ax.set_title(f"Assistant arm ({sim.backend})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "trajectory.png", dpi=120)
    plt.close(fig)
    print(json.dumps({"backend": sim.backend, "steps": len(traj)}))


if __name__ == "__main__":
    main()
