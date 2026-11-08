"""Publish sequential perception as JSON on /surgical/perception.

Replays the inference log at --rate. Two fault injections exist for testing
the assistant: --stall-after N goes silent for --stall-seconds after frame N,
and --swap-every K sends every K-th pair of frames in reverse order.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def publish_order(n: int, swap_every: int) -> list[int]:
    order = list(range(n))
    if swap_every:
        for i in range(swap_every, n - 1, swap_every):
            order[i], order[i + 1] = order[i + 1], order[i]
    return order


def main() -> None:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=20)

    p = argparse.ArgumentParser()
    p.add_argument("--replay", default="evaluation/inference/predictions.jsonl")
    p.add_argument("--rate", type=float, default=15.0)
    p.add_argument("--stall-after", type=int, default=0)
    p.add_argument("--stall-seconds", type=float, default=2.0)
    p.add_argument("--swap-every", type=int, default=0)
    args = p.parse_args()
    path = Path(args.replay)
    if not path.exists():
        raise SystemExit(f"missing {path}. Run python -m surgisense infer first.")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    rclpy.init()
    node = Node("perception", enable_rosout=False, start_parameter_services=False)
    pub = node.create_publisher(String, "/surgical/perception", qos)
    for _ in range(40):
        rclpy.spin_once(node, timeout_sec=0.05)
        if pub.get_subscription_count() > 0:
            break
    print(f"publishing {len(rows)} frames to {pub.get_subscription_count()} subscribers", flush=True)
    period = 1.0 / args.rate
    for sent, i in enumerate(publish_order(len(rows), args.swap_every)):
        if args.stall_after and sent == args.stall_after:
            print(f"stalling {args.stall_seconds}s after {sent} frames", flush=True)
            time.sleep(args.stall_seconds)
        msg = String()
        msg.data = json.dumps({**rows[i]["state"], "published_at": time.time()})
        pub.publish(msg)
        time.sleep(period)
    node.destroy_node()
    print("done", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
