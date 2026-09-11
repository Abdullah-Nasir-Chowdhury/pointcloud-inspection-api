"""Build banks and evaluate every category sequentially, appending to results/run_all.log.

    python scripts/run_all.py --skip bagel
Each category's metrics are merged into results/metrics.json as it finishes, so a crash
half-way keeps what was done. Re-running skips categories that already have a bank unless
--rebuild is given.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from pcinspect.data.mvtec3d import CATEGORIES

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    cats = [c for c in (args.only or CATEGORIES) if c not in args.skip]
    results_path = ROOT / "results" / "metrics.json"
    log = ROOT / "results" / "run_all.log"
    log.parent.mkdir(exist_ok=True)

    for c in cats:
        t0 = time.time()
        if args.rebuild or not (ROOT / "models" / c / "bank.npz").exists():
            subprocess.run([PY, "scripts/build_bank.py", "--categories", c], cwd=ROOT, check=True)
        # evaluate.py writes results/metrics.json for the categories it was given; merge.
        per_cat = ROOT / "results" / f"metrics_{c}.json"
        subprocess.run([PY, "scripts/evaluate.py", "--categories", c, "--out", str(per_cat.parent / f"tmp_{c}")],
                       cwd=ROOT, check=True)
        r = json.loads((per_cat.parent / f"tmp_{c}" / "metrics.json").read_text())[c]
        merged = json.loads(results_path.read_text()) if results_path.exists() else {}
        merged.pop("mean", None)
        merged[c] = r
        results_path.write_text(json.dumps(merged, indent=2))
        with log.open("a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {c}: I-AUROC={r['image_auroc']:.3f} "
                    f"AUPRO={r['aupro_30']:.3f} lat={r['latency_ms_median']:.0f}ms ({time.time()-t0:.0f}s)\n")
        print(open(log).read().splitlines()[-1], flush=True)


if __name__ == "__main__":
    main()
