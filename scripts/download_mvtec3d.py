"""Download and extract MVTec 3D-AD (CC BY-NC-SA 4.0, ~11.6 GB tar.xz).

Usage:
    python scripts/download_mvtec3d.py            # download + extract to data/
    python scripts/download_mvtec3d.py --extract-only path/to/archive.tar.xz

The dataset is released by MVTec Software GmbH for non-commercial research.
Cite: Bergmann et al., "The MVTec 3D-AD Dataset for Unsupervised 3D Anomaly
Detection and Localization", VISAPP 2022.
"""
from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

# Links from https://www.mvtec.com/research-teaching/datasets/mvtec-3d-ad/downloads (2026-09)
URL = (
    "https://www.mydrive.ch/shares/150646/25de33f1276dd418cd856f2f48a84cdb/"
    "download/428824485-1643299897/mvtec_3d_anomaly_detection.tar.xz"
)
_M = "https://www.mydrive.ch/shares/"
CATEGORY_URLS = {
    "bagel": _M + "45891/477cd0a09ea8683b7b77463cc205baa8/download/428793580-1643274607/bagel.tar.xz",
    "cable_gland": _M + "45892/8836f2fd7407ec1c54dc165a5b8a2e91/download/428792309-1643274318/cable_gland.tar.xz",
    "carrot": _M + "45893/4d27bf0566fbd64f05aa34334324b44a/download/428796332-1643275849/carrot.tar.xz",
    "cookie": _M + "45894/2e94f8ff5f36162f8be266e7b482d1b6/download/428796516-1643276093/cookie.tar.xz",
    "dowel": _M + "45895/4bc9099f7976819d8d458453e831fc7c/download/428793974-1643275086/dowel.tar.xz",
    "foam": _M + "45896/fb990326daa1f3325fa535411bb188c6/download/428797064-1643276186/foam.tar.xz",
    "peach": _M + "45897/055d18562435430a64d1edc98da58d2b/download/428797463-1643276448/peach.tar.xz",
    "potato": _M + "45898/7b07b3c833a309112d6daca71f31cba2/download/428797964-1643276600/potato.tar.xz",
    "rope": _M + "45899/cec7d3a6c2585260ddcc58dd2269e0ee/download/428798351-1643276827/rope.tar.xz",
    "tire": _M + "45900/c8261149763d942e033a22a0e8597b5d/download/428798689-1643276973/tire.tar.xz",
}
ROOT = Path(__file__).resolve().parents[1] / "data"
ARCHIVE = ROOT / "mvtec_3d_anomaly_detection.tar.xz"
TARGET = ROOT / "mvtec_3d"

CATEGORIES = [
    "bagel", "cable_gland", "carrot", "cookie", "dowel",
    "foam", "peach", "potato", "rope", "tire",
]


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    done = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    with urllib.request.urlopen(req) as r:
        total = int(r.headers.get("Content-Length", 0)) + done
        mode = "ab" if done else "wb"
        with open(tmp, mode) as f, tqdm(total=total, initial=done, unit="B", unit_scale=True, desc=dest.name) as bar:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def extract(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:xz") as tf:
        members = tf.getmembers()
        for m in tqdm(members, desc="extract", unit="file"):
            tf.extract(m, target, filter="data")
    # Archive extracts to a single top-level folder; flatten it.
    children = [p for p in target.iterdir() if p.is_dir()]
    if len(children) == 1 and children[0].name not in CATEGORIES:
        inner = children[0]
        for p in inner.iterdir():
            p.rename(target / p.name)
        inner.rmdir()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-only", type=Path, help="skip download, extract this archive")
    ap.add_argument("--keep-archive", action="store_true")
    ap.add_argument("--categories", nargs="+", choices=CATEGORIES,
                    help="download only these per-category archives instead of the full set")
    args = ap.parse_args()

    if args.categories:
        for c in args.categories:
            if (TARGET / c).exists():
                print(f"{c}: already extracted")
                continue
            arc = ROOT / f"{c}.tar.xz"
            if not arc.exists():
                download(CATEGORY_URLS[c], arc)
            extract(arc, TARGET)
            if not args.keep_archive:
                arc.unlink()
        wanted = args.categories
    else:
        archive = args.extract_only or ARCHIVE
        if not args.extract_only:
            if archive.exists():
                print(f"archive already present: {archive}")
            else:
                download(URL, archive)
        if TARGET.exists() and all((TARGET / c).exists() for c in CATEGORIES):
            print(f"dataset already extracted at {TARGET}")
            return 0
        extract(archive, TARGET)
        if not args.keep_archive and not args.extract_only:
            archive.unlink()
        wanted = CATEGORIES
    missing = [c for c in wanted if not (TARGET / c).exists()]
    if missing:
        print("missing categories after extract:", missing, file=sys.stderr)
        return 1
    print(f"ok: {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
