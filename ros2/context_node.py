"""Subscribe to perception, add uncertainty, events, and history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from surgisense.context import ContextTracker


def main() -> None:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=20)

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--threshold", type=float)
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    threshold = args.threshold if args.threshold is not None else cfg["uncertainty"]["threshold"]
    thr_path = Path("evaluation/threshold.json")
    if args.threshold is None and thr_path.exists():
        threshold = json.loads(thr_path.read_text())["threshold"]

    rclpy.init()
    node = Node("context", enable_rosout=False, start_parameter_services=False)
    pub = node.create_publisher(String, "/surgical/context", qos)
    tracker = ContextTracker(threshold)

    def on_msg(msg: String) -> None:
        state = tracker.update(json.loads(msg.data))
        out = String()
        out.data = json.dumps(state.to_dict())
        pub.publish(out)
        print(f"context frame {state.frame} uncertain={state.uncertain} conf={state.confidence:.2f}", flush=True)

    node.create_subscription(String, "/surgical/perception", on_msg, qos)
    from rclpy.executors import ExternalShutdownException

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
