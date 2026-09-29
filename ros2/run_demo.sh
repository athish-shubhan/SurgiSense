#!/bin/bash
# Three ROS 2 processes. Perception replays the trained inference log.
# Context and the assistant communicate only through topics.
set -eo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .rosenv/bin/python ]]; then
  echo "ROS 2 env missing at .rosenv. See docs/architecture.md." >&2
  exit 1
fi
if [[ -f .rosenv/setup.bash ]]; then
  # setup.bash reads optional variables that are unset in a nounset shell.
  set +u
  # shellcheck disable=SC1091
  source .rosenv/setup.bash
  set -u
fi
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://${PWD}/ros2/cyclonedds.xml"
unset ROS_LOCALHOST_ONLY || true
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"
PY=".rosenv/bin/python"
RATE="${1:-15}"
"$PY" ros2/context_node.py &
CTX=$!
"$PY" ros2/assistant_node.py &
ARM=$!
sleep 3
"$PY" ros2/perception_node.py --rate "$RATE" &
PERC=$!
cleanup() {
  kill "$PERC" "$CTX" "$ARM" 2>/dev/null || true
  wait "$PERC" "$CTX" "$ARM" 2>/dev/null || true
}
trap cleanup EXIT
wait "$PERC" || true
sleep 2
cleanup
trap - EXIT
echo "assistant log: evaluation/sim_frames/ros_trajectory.json"
