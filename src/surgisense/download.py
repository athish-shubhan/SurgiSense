"""Download the public SurgVU archives. Does not fetch the 344 GB video set."""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

PUBLIC = {
    "cat1_test_set_public.zip": "https://storage.googleapis.com/isi-surgvu/cat1_test_set_public.zip",
    "surgvu24_labels.zip": "https://storage.googleapis.com/isi-surgvu/surgvu24_labels_updated_v2.zip",
}
FULL_VIDEOS = "https://storage.googleapis.com/isi-surgvu/surgvu24_videos_only.zip"
FULL_BYTES = 344_247_725_367


def _fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"exists {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Download public SurgVU files")
    p.add_argument("--raw", default="data/raw")
    args = p.parse_args()
    raw = Path(args.raw)
    raw.mkdir(parents=True, exist_ok=True)
    cat1_zip = raw / "cat1_test_set_public.zip"
    label_zip = raw / "surgvu24_labels.zip"
    _fetch(PUBLIC["cat1_test_set_public.zip"], cat1_zip)
    _fetch(PUBLIC["surgvu24_labels.zip"], label_zip)
    cat1 = raw / "cat1"
    if not (cat1 / "1_fps1.mp4").exists():
        print("extracting tool clips")
        with zipfile.ZipFile(cat1_zip) as zf:
            zf.extractall(cat1)
    labels = raw / "labels"
    if not labels.exists():
        print("extracting case label tables")
        with zipfile.ZipFile(label_zip) as zf:
            zf.extractall(raw)
    print(f"skipping {FULL_VIDEOS} ({FULL_BYTES / 1e9:.0f} GB)")
    print("ready", cat1)


if __name__ == "__main__":
    main()
