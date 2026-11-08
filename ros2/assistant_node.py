"""Fixed-rate arm control loop. Perception only updates the target.

The arm steps on its own 50 Hz timer, as a real controller would. Each tick
it tracks the latest accepted command, or holds when that command is
uncertain, when no context has arrived for --stale-after seconds, or before
the first message. Reordered and duplicate frames are dropped.
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np

from surgisense.context import HOLD, Watchdog, assistant_command
from surgisense.sim import ArmSim


def main() -> None:
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    p = argparse.ArgumentParser()
    p.add_argument("--stale-after", type=float, default=0.25)
    p.add_argument("--control-hz", type=float, default=50.0)
    p.add_argument("--log", default="evaluation/ros/assistant_log.json")
    args = p.parse_args()

    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=20)
    rclpy.init()
    node = Node("assistant", enable_rosout=False, start_parameter_services=False)
    pub = node.create_publisher(String, "/surgical/assistant", qos)
    sim = ArmSim()
    period = 1.0 / args.control_hz
    substeps = max(1, round(period / sim.timestep))
    watchdog = Watchdog(timeout_s=args.stale_after)
    command = {**HOLD, "reason": "startup", "frame": None}
    ticks: list[dict] = []
    latency_ms: list[float] = []
    stale_events: list[dict] = []
    t0 = time.monotonic()
    context_rejected = 0
    accepted = 0
    node.get_logger().info(f"arm backend {sim.backend}, {args.control_hz:.0f} Hz, stale after {args.stale_after}s")

    def on_msg(msg: String) -> None:
        nonlocal command, context_rejected, accepted
        state = json.loads(msg.data)
        context_rejected = max(context_rejected, int(state.get("context_rejected") or 0))
        if not watchdog.accept(state["video"], state["frame"], time.monotonic()):
            return
        accepted += 1
        if "published_at" in state:
            latency_ms.append((time.time() - state["published_at"]) * 1000)
        cmd = assistant_command(state, 0.0)  # context already applied the threshold
        command = {**cmd, "frame": state["frame"]}

    def tick() -> None:
        now = time.monotonic()
        cmd = command
        if watchdog.last_seen is not None and watchdog.stale(now):
            if ticks and ticks[-1]["reason"] != "stale":
                stale_events.append(
                    {"t": now - t0, "frame": watchdog.last_frame, "silent_ms": (now - watchdog.last_seen) * 1000}
                )
            cmd = {**HOLD, "reason": "stale", "frame": command["frame"]}
        q = sim.step(cmd["joints"], n=substeps)
        row = {"t": now - t0, "frame": cmd["frame"], "mode": cmd["mode"], "reason": cmd["reason"], "q": [float(v) for v in q]}
        ticks.append(row)
        out = String()
        out.data = json.dumps(row)
        pub.publish(out)

    node.create_subscription(String, "/surgical/context", on_msg, qos)
    node.create_timer(period, tick)

    def finish(*_) -> None:
        raise SystemExit(0)  # the finally block below writes the log once

    signal.signal(signal.SIGTERM, finish)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit, ExternalShutdownException):
        pass
    finally:
        write_log(Path(args.log), sim.backend, args, ticks, latency_ms, stale_events, accepted, watchdog.rejected + context_rejected)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def write_log(path, backend, args, ticks, latency_ms, stale_events, accepted, rejected) -> None:
    q = np.asarray([t["q"] for t in ticks]) if ticks else np.zeros((0, 2))
    moved = np.abs(np.diff(q, axis=0)).max(axis=1) if len(q) > 1 else np.zeros(0)
    # Motion on a tick is attributed to that tick's command.
    hold_motion = [float(m) for m, t in zip(moved, ticks[1:]) if t["mode"] == "hold"]
    reasons: dict[str, int] = {}
    for t in ticks:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
    lat = np.asarray(latency_ms)
    summary = {
        "backend": backend,
        "control_hz": args.control_hz,
        "stale_after_s": args.stale_after,
        "ticks": len(ticks),
        "ticks_by_reason": reasons,
        "frames_accepted": accepted,
        "frames_rejected_out_of_order": rejected,
        "max_joint_change_on_hold_ticks": max(hold_motion, default=0.0),
        "stale_events": stale_events,
        "perception_to_assistant_ms": {
            "n": int(len(lat)),
            "p50": float(np.percentile(lat, 50)) if len(lat) else None,
            "p95": float(np.percentile(lat, 95)) if len(lat) else None,
            "p99": float(np.percentile(lat, 99)) if len(lat) else None,
            "max": float(lat.max()) if len(lat) else None,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary, "ticks": ticks}))
    path.with_name(path.stem + "_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
