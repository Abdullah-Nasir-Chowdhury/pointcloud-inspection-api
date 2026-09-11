"""Hyper-parameter sweep on one category: fit + calibrate + evaluate per setting.

    python scripts/sweep.py --category cable_gland --target-points 0 3000 6000 12000 --max-train 120
target-points 0 means the fixed 2 mm voxel. Models go to runs/sweep/<category>/<tag>/ and
the table to results/sweep_<category>.md.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

from pcinspect.data.mvtec3d import list_samples
from pcinspect.inspector import Inspector, InspectorConfig
from pcinspect.preprocess import PreprocessConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import evaluate_category  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--target-points", nargs="+", type=int, default=[0, 3000, 6000, 12000])
    ap.add_argument("--topk", nargs="+", type=float, default=[0.01])
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "mvtec_3d")
    args = ap.parse_args()

    cat = args.category
    train = [s.load_xyz() for s in list_samples(args.data, cat, "train")[: args.max_train]]
    val = [s.load_xyz() for s in list_samples(args.data, cat, "validation")]
    rows = []
    for tp in args.target_points:
        for topk in args.topk:
            tag = f"tp{tp}_topk{topk}"
            t0 = time.time()
            cfg = InspectorConfig(pre=PreprocessConfig(voxel_size=0.002), target_points=tp or None,
                                  topk_fraction=topk)
            insp = Inspector.fit(train, cfg)
            insp.calibrate(val)
            out = ROOT / "runs" / "sweep" / cat / tag
            insp.save(out / cat)
            r = evaluate_category(args.data, out, cat)
            rows.append((tag, insp.cfg.pre.voxel_size * 1000, r, time.time() - t0))
            print(f"{tag}: voxel={insp.cfg.pre.voxel_size*1000:.2f}mm I-AUROC={r['image_auroc']:.3f} "
                  f"P-AUROC={r['pixel_auroc']:.3f} AUPRO={r['aupro_30']:.3f} lat={r['latency_ms_median']:.0f}ms "
                  f"({time.time()-t0:.0f}s)", flush=True)

    lines = [f"# Sweep: {cat}", "", "| Setting | Voxel (mm) | I-AUROC | P-AUROC | AUPRO@30% | Acc@thr | Latency (ms) | Time (s) |",
             "|---|---|---|---|---|---|---|---|"]
    for tag, v, r, t in rows:
        lines.append(f"| {tag} | {v:.2f} | {r['image_auroc']:.3f} | {r['pixel_auroc']:.3f} | {r['aupro_30']:.3f} | "
                     f"{r['accuracy_at_threshold']:.3f} | {r['latency_ms_median']:.0f} | {t:.0f} |")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"sweep_{cat}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
