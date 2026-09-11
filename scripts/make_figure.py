"""Render README figures: RGB | depth | anomaly overlay | ground truth, one row per defect type.

    python scripts/make_figure.py --category bagel            -> assets/bagel.png
    python scripts/make_figure.py --all                       -> one figure per category
Picks the first test scan of each defect type (plus one good scan) and crops to the object.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pcinspect.data.mvtec3d import CATEGORIES, list_samples
from pcinspect.inspector import Inspector

ROOT = Path(__file__).resolve().parents[1]


def depth_gray(xyz: np.ndarray, obj: np.ndarray | None = None) -> np.ndarray:
    """Depth as grey, contrast stretched over the object (not the table) so shape is visible."""
    z = xyz[..., 2]; valid = (xyz != 0).any(-1)
    ref = valid & obj if obj is not None and (valid & obj).any() else valid
    lo, hi = np.percentile(z[ref], [1, 99])
    g = (255 * (1 - np.clip((z - lo) / max(hi - lo, 1e-9), 0, 1))).astype(np.uint8)
    g[~valid] = 0
    return np.stack([g] * 3, -1)


def colorize(heat: np.ndarray, vmax: float) -> np.ndarray:
    t = np.clip(heat / max(vmax, 1e-9), 0, 1)
    r = np.clip(3 * t, 0, 1); g = np.clip(3 * t - 1, 0, 1); b = np.clip(np.where(t < 0.5, 1.5 * t, 3 * t - 2), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def overlay(rgb: np.ndarray, heat: np.ndarray, vmax: float) -> np.ndarray:
    a = np.clip(heat / max(vmax, 1e-9), 0, 1)[..., None] * 0.8
    return (rgb * (1 - a) + colorize(heat, vmax) * a).astype(np.uint8)


def crop_box(mask: np.ndarray, pad: int = 30) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    H, W = mask.shape
    return max(ys.min() - pad, 0), min(ys.max() + pad, H), max(xs.min() - pad, 0), min(xs.max() + pad, W)


def render(category: str, tile: int = 220) -> Path:
    insp = Inspector.load(ROOT / "models" / category)
    test = list_samples(ROOT / "data" / "mvtec_3d", category, "test")
    picks = [next(s for s in test if not s.is_anomalous)]
    for d in sorted({s.defect for s in test if s.is_anomalous}):
        picks.append(next(s for s in test if s.defect == d))
    vmax = insp.threshold * 1.5
    rows = []
    for s in picks:
        xyz = s.load_xyz(); rgb = s.load_rgb(); gt = s.load_gt()
        r = insp.inspect(xyz)
        obj = r.heatmap > 0
        y0, y1, x0, x1 = crop_box(obj)
        gt_img = np.zeros_like(rgb); gt_img[..., 0] = 255 * (gt if gt is not None else 0)
        panels = [rgb, depth_gray(xyz, obj), overlay(rgb, r.heatmap, vmax), gt_img]
        tiles = [Image.fromarray(p[y0:y1, x0:x1]).resize((tile, tile)) for p in panels]
        row = Image.new("RGB", (tile * 4 + 130, tile), "white")
        for i, t in enumerate(tiles):
            row.paste(t, (130 + i * tile, 0))
        d = ImageDraw.Draw(row)
        label = f"{s.defect}\nscore {r.image_score:.1f}\n{'FAIL' if r.is_defective else 'PASS'}"
        d.multiline_text((8, tile // 2 - 24), label, fill=(0, 0, 0))
        rows.append(row)
    header = Image.new("RGB", (rows[0].width, 22), "white")
    d = ImageDraw.Draw(header)
    for i, name in enumerate(["RGB", "depth", "anomaly heatmap", "ground truth"]):
        d.text((130 + i * tile + 6, 5), name, fill=(0, 0, 0))
    fig = Image.new("RGB", (rows[0].width, 22 + sum(r.height for r in rows)), "white")
    fig.paste(header, (0, 0))
    y = 22
    for r in rows:
        fig.paste(r, (0, y)); y += r.height
    out = ROOT / "assets" / f"{category}.png"
    out.parent.mkdir(exist_ok=True)
    fig.save(out, optimize=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="bagel")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    cats = [c for c in CATEGORIES if (ROOT / "models" / c / "bank.npz").exists()] if args.all else [args.category]
    for c in cats:
        print(render(c))


if __name__ == "__main__":
    main()
