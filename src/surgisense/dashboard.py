"""Local dashboard for a finished inference run."""

from __future__ import annotations

import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def build(out: Path, inference: Path) -> Path:
    summary_path = inference / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"missing {summary_path}. Run inference first.")
    summary = json.loads(summary_path.read_text())
    rows = [json.loads(line) for line in (inference / "predictions.jsonl").read_text().splitlines() if line.strip()]
    frames = sorted((inference / "frames").glob("*.jpg"))
    cards = []
    for frame in frames:
        idx = int(frame.stem)
        row = next((r for r in rows if r["state"]["frame"] == idx), None)
        if row is None:
            continue
        state, cmd = row["state"], row["command"]
        tools = ", ".join(f"{t['name']} {t['score']:.2f}" for t in state["tools"][:4]) or "none"
        cards.append(
            f"<figure><img src='frames/{frame.name}'><figcaption>"
            f"frame {idx} · {cmd['mode']} · conf {state['confidence']:.2f}<br>{tools}"
            f"<br>{state['latency_ms'].get('total_ms', 0):.1f} ms</figcaption></figure>"
        )
    # The page is served with inference/ as the root, so image paths stay relative.
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SurgiSense</title>
<style>
body {{ font: 15px/1.4 -apple-system, sans-serif; margin: 24px; color: #1c1c1c; background: #f6f4ef; }}
h1 {{ font-size: 22px; margin-bottom: 4px; }}
.note {{ color: #6b3a3a; }}
dl {{ display: grid; grid-template-columns: 180px 1fr; max-width: 640px; }}
dt {{ color: #555; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }}
figure {{ margin: 0; background: white; padding: 8px; }}
img {{ width: 100%; height: auto; }}
figcaption {{ font-size: 13px; margin-top: 6px; }}
</style></head><body>
<h1>SurgiSense</h1>
<p>Sequential predictions on {summary['video']}. Research simulation only. Not for clinical use.</p>
<dl>
<dt>device</dt><dd>{summary['device']}</dd>
<dt>resolution</dt><dd>{summary['resolution'][0]}×{summary['resolution'][1]}, batch {summary['batch_size']}</dd>
<dt>preprocess</dt><dd>{summary['preprocess_ms']:.1f} ms</dd>
<dt>detector</dt><dd>{summary['detector_ms']:.1f} ms</dd>
<dt>temporal</dt><dd>{summary['temporal_ms']:.1f} ms</dd>
<dt>postprocess</dt><dd>{summary['post_ms']:.1f} ms</dd>
<dt>total</dt><dd>{summary['total_ms']:.1f} ms ({summary['fps']:.2f} FPS)</dd>
<dt>hold / track</dt><dd>{summary['hold_frames']} / {summary['track_frames']}</dd>
</dl>
<p class="note">HOLD means confidence is below the validation threshold, so the assistant does not update its pose.</p>
<div class="grid">{''.join(cards)}</div>
</body></html>"""
    page = out / "index.html"
    page.write_text(html)
    return page


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Build and optionally serve the dashboard")
    p.add_argument("--inference", default="evaluation/inference")
    p.add_argument("--serve", action="store_true")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    inference = Path(args.inference)
    page = build(inference, inference)
    print(page)
    if args.serve:
        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *a, **k):
                super().__init__(*a, directory=str(inference), **k)

        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
        print(f"http://127.0.0.1:{args.port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
