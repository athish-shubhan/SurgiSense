"""Tool state and the hold/track rule for the assistant arm."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


# Joint targets are assistant postures, not surgical motions.
POSES = {
    "grasping retractor": (0.4, -0.6),
    "cadiere forceps": (0.5, 0.8),
    "bipolar forceps": (-0.4, 0.9),
    "force bipolar": (-0.2, 1.1),
    "clip applier": (0.8, 0.3),
    "monopolar curved scissor": (-0.7, 0.5),
    "vessel sealer": (0.2, 1.2),
    "needle driver": (0.6, 0.7),
}


@dataclass
class SurgicalState:
    timestamp: float
    video: str
    frame: int
    tools: list[dict]
    presence: dict
    confidence: float
    uncertain: bool
    events: list[str] = field(default_factory=list)
    history: list[list[str]] = field(default_factory=list)
    latency_ms: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def tool_names(tools: list[dict]) -> set[str]:
    return {t["name"] for t in tools}


def events_from(prev_names: set[str], curr_names: set[str], prev_uncertain: bool, uncertain: bool) -> list[str]:
    events = [f"entered:{name}" for name in sorted(curr_names - prev_names)]
    events += [f"exited:{name}" for name in sorted(prev_names - curr_names)]
    if uncertain and not prev_uncertain:
        events.append("became_uncertain")
    if (not uncertain) and prev_uncertain:
        events.append("became_confident")
    return events


def build_state(
    timestamp: float,
    video: str,
    frame: int,
    tools: list[dict],
    presence: dict[str, float],
    threshold: float,
    history: list[list[str]] | None = None,
    prev_uncertain: bool = True,
    latency_ms: dict | None = None,
) -> SurgicalState:
    tools = sorted(tools, key=lambda t: -t["score"])
    confidence = tools[0]["score"] if tools else 0.0
    if presence and tools:
        confidence = min(confidence, float(presence.get(tools[0]["name"], 0.0)))
    uncertain = confidence < threshold
    prev = set(history[-1]) if history else set()
    events = events_from(prev, tool_names(tools), prev_uncertain, uncertain)
    hist = list(history or [])
    hist.append(sorted(tool_names(tools)))
    return SurgicalState(
        timestamp=timestamp,
        video=video,
        frame=frame,
        tools=tools,
        presence=presence,
        confidence=confidence,
        uncertain=uncertain,
        events=events,
        history=hist[-8:],
        latency_ms=latency_ms or {},
    )


def assistant_command(state: dict, threshold: float) -> dict:
    if state["uncertain"] or float(state["confidence"]) < threshold or not state["tools"]:
        return {"mode": "hold", "tool": None, "joints": None, "reason": "uncertain"}
    tool = state["tools"][0]["name"]
    pose = POSES.get(tool, (0.0, 0.6))
    return {"mode": "track", "tool": tool, "joints": [float(pose[0]), float(pose[1])], "reason": "confident"}


class ContextTracker:
    """The state the context node keeps between perception messages.

    `update` turns one perception message (a `SurgicalState` dict as published by the
    perception node) into the next state, carrying history and the previous uncertainty
    flag forward. The ROS node and the offline replay test both go through this.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.history: list[list[str]] = []
        self.prev_uncertain = True

    def update(self, raw: dict) -> SurgicalState:
        state = build_state(
            raw["timestamp"],
            raw["video"],
            raw["frame"],
            raw["tools"],
            raw["presence"],
            self.threshold,
            self.history,
            self.prev_uncertain,
            raw.get("latency_ms"),
        )
        self.prev_uncertain = state.uncertain
        self.history = state.history
        return state
