import json

from surgisense.replay import frame_view, load_log, render, tool_colours


def record(frame, mode, confidence, tools, tool=None):
    return {
        "state": {"frame": frame, "confidence": confidence, "tools": tools, "presence": {"needle driver": 0.9, "clip applier": 0.1}},
        "command": {"mode": mode, "tool": tool, "joints": [0.0, 0.0], "reason": "confident" if mode == "track" else "uncertain"},
    }


def box(name, score):
    return {"name": name, "score": score, "box": [10.0, 20.0, 110.0, 90.0]}


def test_frame_view_drops_weak_boxes_and_draws_the_strongest_last():
    v = frame_view(record(3, "track", 0.6, [box("needle driver", 0.6), box("clip applier", 0.05), box("clip applier", 0.2)], "needle driver"))
    assert [t["score"] for t in v["boxes"]] == [0.2, 0.6]
    assert (v["frame"], v["mode"], v["tool"], v["confidence"]) == (3, "track", "needle driver", 0.6)


def test_hold_frames_have_no_tool():
    v = frame_view(record(4, "hold", 0.2, []))
    assert v["mode"] == "hold" and v["tool"] is None and v["boxes"] == []


def test_colours_are_stable_and_cover_detected_and_presence_tools():
    records = [record(0, "track", 0.6, [box("vessel sealer", 0.5)], "vessel sealer")]
    colours = tool_colours(records)
    assert set(colours) == {"vessel sealer", "needle driver", "clip applier"}
    assert colours == tool_colours(list(reversed(records)))


def test_render_writes_one_gif_frame_per_log_frame(tmp_path):
    log = tmp_path / "predictions.jsonl"
    rows = [record(i, "track" if i % 2 else "hold", 0.5 if i % 2 else 0.2, [box("needle driver", 0.5)], "needle driver" if i % 2 else None)
            for i in range(6)]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "replay.gif"
    assert render(load_log(log), 0.35, start=1, count=4, out=out, fps=10, history=5) == 4

    from PIL import Image
    with Image.open(out) as gif:
        assert gif.n_frames == 4
