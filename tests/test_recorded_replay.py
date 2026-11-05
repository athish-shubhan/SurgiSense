"""Replay the committed inference log through the context, assistant and arm code.

`evaluation/inference/predictions.jsonl` is what the ROS perception node publishes, and
`evaluation/sim_frames/ros_trajectory.json` is what the assistant node logged on the
recorded run. Replaying offline, with no model and no ROS, must reproduce the README's
hold/track counts and the logged arm trajectory.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from surgisense.context import ContextTracker, assistant_command
from surgisense.sim import ArmSim

ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS = ROOT / "evaluation/inference/predictions.jsonl"
ROS_LOG = ROOT / "evaluation/sim_frames/ros_trajectory.json"
THRESHOLD = ROOT / "evaluation/threshold.json"


@pytest.fixture(scope="module")
def replay():
    rows = [json.loads(line) for line in PREDICTIONS.read_text().splitlines() if line.strip()]
    tracker = ContextTracker(json.loads(THRESHOLD.read_text())["threshold"])
    sim = ArmSim(backend="mujoco")
    out = []
    for row in rows:
        state = tracker.update(row["state"]).to_dict()  # context node
        cmd = assistant_command(state, 0.0)  # assistant node: context already applied the threshold
        out.append({"cmd": cmd, "joints": sim.step(cmd["joints"]), "recorded": row["command"]})
    return out


def test_replay_reproduces_readme_hold_track_counts(replay):
    modes = [r["cmd"]["mode"] for r in replay]
    assert len(modes) == 871
    assert (modes.count("track"), modes.count("hold")) == (417, 454)


def test_replay_matches_commands_logged_by_infer(replay):
    assert [r["cmd"] for r in replay] == [r["recorded"] for r in replay]


def test_replay_matches_logged_ros_trajectory(replay):
    logged = json.loads(ROS_LOG.read_text())
    assert logged["backend"] == "mujoco"
    assert [t["mode"] for t in logged["trajectory"]] == [r["cmd"]["mode"] for r in replay]
    joints = np.array([t["joints"] for t in logged["trajectory"]])
    np.testing.assert_allclose(np.array([r["joints"] for r in replay]), joints, atol=1e-9)


def test_hold_frames_never_move_the_arm(replay):
    for prev, cur in zip(replay, replay[1:]):
        if cur["cmd"]["mode"] == "hold":
            np.testing.assert_array_equal(cur["joints"], prev["joints"])
