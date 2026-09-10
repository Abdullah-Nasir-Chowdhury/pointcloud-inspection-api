"""Evaluate saved banks on the MVTec 3D-AD test split.

    python scripts/evaluate.py --categories bagel
Prints a per-category table and writes results/metrics.json + results/metrics.md.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pcinspect.data.mvtec3d import CATEGORIES, list_samples
from pcinspect.inspector import Inspector
from pcinspect.metrics import aupro, image_auroc, pixel_auroc

ROOT = Path(__file__).resolve().parents[1]


def evaluate_category(data: Path, models: Path, cat: str) -> dict:
    insp = Inspector.load(models / cat)
    test = list_samples(data, cat, "test")
    labels, scores, gts, maps, lat = [], [], [], [], []
    for s in tqdm(test, desc=f"{cat}: test"):
        xyz = s.load_xyz()
        t0 = time.perf_counter()
        r = insp.inspect(xyz)
        lat.append(time.perf_counter() - t0)
        labels.append(s.is_anomalous)
        scores.append(r.image_score)
        gt = s.load_gt()
        gts.append(np.zeros(r.heatmap.shape, bool) if gt is None else gt)
        maps.append(r.heatmap)
    labels, scores = np.asarray(labels), np.asarray(scores)
    pred = scores > insp.threshold
    anomalous = [(g, m) for g, m, l in zip(gts, maps, labels) if l]
    return {
        "n_test": int(len(test)),
        "image_auroc": image_auroc(labels, scores),
        "pixel_auroc": pixel_auroc(gts, maps),
        "aupro_30": aupro([g for g, _ in anomalous], [m for _, m in anomalous]),
        "accuracy_at_threshold": float((pred == labels).mean()),
        "recall_at_threshold": float(pred[labels].mean()),
        "false_alarm_rate": float(pred[~labels].mean()),
        "latency_ms_median": float(np.median(lat) * 1000),
        "bank_size": len(insp.bank),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "mvtec_3d")
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES), choices=CATEGORIES)
    args = ap.parse_args()

    results = {c: evaluate_category(args.data, args.models, c) for c in args.categories}
    keys = ["image_auroc", "pixel_auroc", "aupro_30", "accuracy_at_threshold", "latency_ms_median"]
    results["mean"] = {k: float(np.mean([results[c][k] for c in args.categories])) for k in keys}

    args.out.mkdir(exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2))
    lines = ["| Category | I-AUROC | P-AUROC | AUPRO@30% | Acc@thr | Latency (ms) |", "|---|---|---|---|---|---|"]
    for c, r in results.items():
        lines.append(f"| {c} | {r['image_auroc']:.3f} | {r['pixel_auroc']:.3f} | {r['aupro_30']:.3f} | "
                     f"{r['accuracy_at_threshold']:.3f} | {r['latency_ms_median']:.0f} |")
    table = "\n".join(lines)
    (args.out / "metrics.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
