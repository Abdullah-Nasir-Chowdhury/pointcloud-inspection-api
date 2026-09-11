"""Per-stage latency benchmark on real scans (CPU).

    python scripts/benchmark.py --category bagel --n 20
Prints median ms for: load, preprocess, FPFH, NN search, heatmap, total; plus hardware info.
Results go to results/benchmark_<category>.md for the README.
"""
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from pcinspect.data.mvtec3d import list_samples
from pcinspect.features.fpfh import compute_fpfh
from pcinspect.inspector import Inspector
from pcinspect.preprocess import preprocess

ROOT = Path(__file__).resolve().parents[1]


def cpu_name() -> str:
    try:
        import subprocess
        out = subprocess.check_output(["wmic", "cpu", "get", "name"], text=True).splitlines()
        return next(l.strip() for l in out[1:] if l.strip())
    except Exception:  # noqa: BLE001
        return platform.processor() or platform.machine()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="bagel")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "mvtec_3d")
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    args = ap.parse_args()

    insp = Inspector.load(args.models / args.category)
    cfg = insp.cfg
    samples = list_samples(args.data, args.category, "test")[: args.n]
    stages = {k: [] for k in ("load", "preprocess", "fpfh", "search", "heatmap", "total")}
    for s in samples:
        t = [time.perf_counter()]
        xyz = s.load_xyz(); t.append(time.perf_counter())
        pre = preprocess(xyz, cfg.pre); t.append(time.perf_counter())
        feats = compute_fpfh(pre.down_points, pre.down_normals, cfg.pre.voxel_size, cfg.fpfh); t.append(time.perf_counter())
        d = insp.bank.score(feats); t.append(time.perf_counter())
        H, W = pre.shape
        heat = np.zeros(H * W, np.float32); heat[pre.pix_idx] = d[pre.down_to_full]
        gaussian_filter(heat.reshape(H, W), cfg.heatmap_sigma); t.append(time.perf_counter())
        for k, (a, b) in zip(("load", "preprocess", "fpfh", "search", "heatmap"), zip(t[:-1], t[1:])):
            stages[k].append((b - a) * 1000)
        stages["total"].append((t[-1] - t[1]) * 1000)   # excludes disk load

    med = {k: float(np.median(v)) for k, v in stages.items()}
    lines = [f"CPU: {cpu_name()} | bank size: {len(insp.bank):,} | scans: {len(samples)} | scan size: {H}x{W}", "",
             "| Stage | Median ms |", "|---|---|"]
    lines += [f"| {k} | {v:.0f} |" for k, v in med.items()]
    table = "\n".join(lines)
    out = ROOT / "results" / f"benchmark_{args.category}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
