"""Subscribe to context and step the assistant arm. Uncertainty holds the arm."""

from __future__ import annotations

import json
import signal
from pathlib import Path

from surgisense.context import assistant_command
from surgisense.sim import ArmSim


def main() -> None:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=20)

    rclpy.init()
    node = Node("assistant", enable_rosout=False, start_parameter_services=False)
    pub = node.create_publisher(String, "/surgical/assistant", qos)
    sim = ArmSim()
    log = []
    node.get_logger().info(f"arm backend {sim.backend}")

    def on_msg(msg: String) -> None:
        state = json.loads(msg.data)
        # Context already set `uncertain` from the validation threshold, so no second threshold here:
        # this holds when the state is uncertain or has no tools, and tracks the top tool otherwise.
        cmd = assistant_command(state, 0.0)
        q = sim.step(cmd["joints"])
        out = {
            "frame": state["frame"],
            "mode": cmd["mode"],
            "tool": cmd["tool"],
            "joints": [float(v) for v in q],
            "confidence": state["confidence"],
            "backend": sim.backend,
        }
        log.append(out)
        print(f"frame {out['frame']} {out['mode']} q={out['joints']}", flush=True)
        msg_out = String()
        msg_out.data = json.dumps(out)
        pub.publish(msg_out)
        _dump(sim.backend, log)

    node.create_subscription(String, "/surgical/context", on_msg, qos)
    from rclpy.executors import ExternalShutdownException

    signal.signal(signal.SIGTERM, lambda *_: (_dump(sim.backend, log), __import__("sys").exit(0)))
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit, ExternalShutdownException):
        pass
    finally:
        _dump(sim.backend, log)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _dump(backend: str, log: list) -> None:
    Path("evaluation/sim_frames").mkdir(parents=True, exist_ok=True)
    Path("evaluation/sim_frames/ros_trajectory.json").write_text(
        json.dumps({"backend": backend, "trajectory": log}, indent=2)
    )


if __name__ == "__main__":
    main()
