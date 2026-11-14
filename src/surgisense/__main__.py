"""python -m surgisense <command>"""

from __future__ import annotations

import sys


def main() -> None:
    commands = {
        "download": "surgisense.download",
        "preprocess": "surgisense.preprocess",
        "explore": "surgisense.explore",
        "train-detector": "surgisense.train_detector",
        "train-temporal": "surgisense.train_temporal",
        "evaluate": "surgisense.evaluate",
        "infer": "surgisense.infer",
        "simulate": "surgisense.simulate",
        "dashboard": "surgisense.dashboard",
        "replay": "surgisense.replay",
    }
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        names = ", ".join(commands)
        raise SystemExit(f"usage: python -m surgisense <command>\ncommands: {names}")
    mod = __import__(commands[sys.argv[1]], fromlist=["main"])
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    mod.main()


if __name__ == "__main__":
    main()
