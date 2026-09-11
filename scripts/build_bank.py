"""Fit one memory bank per MVTec 3D-AD category and calibrate its pass/fail threshold.

    python scripts/build_bank.py --categories bagel            # one category
    python scripts/build_bank.py                               # all ten
Writes models/<category>/{bank.npz,meta.json}.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from tqdm import tqdm

from pcinspect.data.mvtec3d import CATEGORIES, list_samples
from pcinspect.inspector import Inspector, InspectorConfig

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "mvtec_3d")
    ap.add_argument("--out", type=Path, default=ROOT / "models")
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES), choices=CATEGORIES)
    ap.add_argument("--max-train", type=int, default=None, help="cap training scans (for quick runs)")
    ap.add_argument("--percentile", type=float, default=99.0, help="validation percentile for threshold")
    ap.add_argument("--target-points", type=int, default=3000, help="per-category point budget (0 = fixed voxel)")
    ap.add_argument("--voxel", type=float, default=0.002, help="fixed voxel size in m when --target-points 0")
    args = ap.parse_args()

    from pcinspect.preprocess import PreprocessConfig
    cfg = InspectorConfig(pre=PreprocessConfig(voxel_size=args.voxel),
                          target_points=args.target_points or None)
    for cat in args.categories:
        t0 = time.time()
        train = list_samples(args.data, cat, "train")[: args.max_train]
        val = list_samples(args.data, cat, "validation")
        scans = [s.load_xyz() for s in tqdm(train, desc=f"{cat}: load train")]
        insp = Inspector.fit(scans, cfg)
        thr = insp.calibrate([s.load_xyz() for s in tqdm(val, desc=f"{cat}: calibrate")],
                             percentile=args.percentile)
        insp.save(args.out / cat)
        print(f"{cat}: voxel={insp.cfg.pre.voxel_size*1000:.2f}mm bank={len(insp.bank)} "
              f"threshold={thr:.4f} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
