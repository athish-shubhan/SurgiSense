"""Publish sequential perception as JSON on /surgical/perception."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> None:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=20)

    p = argparse.ArgumentParser()
    p.add_argument("--replay", default="evaluation/inference/predictions.jsonl")
    p.add_argument("--rate", type=float, default=8.0)
    args = p.parse_args()
    path = Path(args.replay)
    if not path.exists():
        raise SystemExit(f"missing {path}. Run python -m surgisense infer first.")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    print(f"publishing {len(rows)} frames", flush=True)
    rclpy.init()
    print("rclpy ready", flush=True)
    node = Node("perception", enable_rosout=False, start_parameter_services=False)
    print("node ready", flush=True)
    pub = node.create_publisher(String, "/surgical/perception", qos)
    print("publisher ready", flush=True)
    for _ in range(40):
        rclpy.spin_once(node, timeout_sec=0.05)
        if pub.get_subscription_count() > 0:
            break
    print(f"subscribers {pub.get_subscription_count()}", flush=True)
    period = 1.0 / args.rate
    for i, row in enumerate(rows):
        msg = String()
        msg.data = json.dumps(row["state"])
        pub.publish(msg)
        print(f"sent {i}", flush=True)
        time.sleep(period)
    node.destroy_node()
    print("done", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
