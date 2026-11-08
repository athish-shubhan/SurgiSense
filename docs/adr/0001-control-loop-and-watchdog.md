# ADR 0001: Fixed-rate control loop with a perception watchdog

Status: accepted

## Context

The assistant node used to step the arm once per incoming `/surgical/context` message. The arm and perception therefore shared one clock. If perception stopped, simulated time stopped too. That made one failure untestable and hid it: on a real controller the arm keeps servoing toward its last target after the vision process dies.

The topics are best effort (`KEEP_LAST 20`). Frames can drop, and nothing prevents a late frame from overwriting a newer one. The context node keeps a tool history and enter/exit events, which assume frames arrive in order.

## Options

1. Keep message-driven stepping and add a timeout check inside the callback. A timeout check that only runs when a message arrives cannot fire when messages stop.
2. Use ROS 2 QoS deadline and liveliness events on the subscription. These are the standard mechanism. Their support differs by RMW and by rclpy version, and this repo runs CycloneDDS on macOS through a conda ROS build. The arm would still need its own clock.
3. Run the arm on its own timer. Messages only update the target. A small watchdog decides each tick whether the latest command can be used.

## Decision

Option 3. `ros2/assistant_node.py` steps the arm at 50 Hz (`--control-hz`). Each MuJoCo step covers one control period (two 10 ms substeps). Each tick the arm:

- holds before the first accepted message,
- holds when the latest context says `uncertain`,
- holds when nothing was accepted for `--stale-after` seconds (0.25 s by default, about four frames at the 15 Hz replay rate),
- otherwise moves toward the posture of the top tool.

`surgisense.context.Watchdog` holds the rule. It is plain Python, so it has unit tests without ROS. Both the context node and the assistant drop a frame unless it is newer than the last accepted frame of the same clip.

## Consequences

- Fault injection is now possible and is measured: `perception_node.py --stall-after N --stall-seconds S` goes silent mid-run, and `--swap-every K` reverses frame pairs. Two recorded runs are in `evaluation/ros/nominal_summary.json` (`bash ros2/run_demo.sh 15`) and `evaluation/ros/stall_summary.json` (`bash ros2/run_demo.sh 15 --stall-after 300 --stall-seconds 2`). Both ran on one Mac with CycloneDDS and MuJoCo. In both, no hold tick moved a joint (`max_joint_change_on_hold_ticks` 0.0). In the stall run the watchdog held for 201 ticks. The nominal run also shows seven short stale events (silence of 250 to 313 ms at a 67 ms frame period), so on this machine the 0.25 s timeout fires on ordinary scheduling jitter; that is a reason to treat the value as a placeholder. The `--swap-every` path is covered by a unit test of the frame order only; no recorded ROS run is committed for it.
- The tick log grows with wall time, not with frames. It is written once at shutdown. The old node rewrote the whole log on every message, which is quadratic in run length.
- 0.25 s is a guess sized to the replay rate, not a clinical number. A real system would derive it from the camera rate and the stopping distance of the arm.
- QoS deadline events (option 2) would complement this and could replace the hand-written timeout on a ROS build where they are reliable. That has not been tested here.
