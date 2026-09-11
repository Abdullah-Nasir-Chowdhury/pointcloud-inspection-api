"""Gradio demo for the point cloud defect inspector (runs on Hugging Face Spaces, CPU).

Upload an organized xyz TIFF (HxWx3 float32, metres) from a structured-light scanner,
pick the part category, and get the anomaly heatmap, score and pass/fail decision.

Local run:  python demo/app.py            (expects ../models/<category>/bank.npz)
"""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np
import tifffile
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))               # local run: import the package from the repo
from pcinspect.inspector import Inspector          # noqa: E402
from pcinspect.preprocess import EmptyScanError    # noqa: E402

MODELS = Path(os.environ.get("PCINSPECT_MODELS", HERE.parent / "models"))
EXAMPLES = HERE / "examples"
_cache: dict[str, Inspector] = {}


def categories() -> list[str]:
    return sorted(p.name for p in MODELS.iterdir() if (p / "bank.npz").exists()) if MODELS.exists() else []


def get(cat: str) -> Inspector:
    if cat not in _cache:
        _cache[cat] = Inspector.load(MODELS / cat)
    return _cache[cat]


def depth_image(xyz: np.ndarray) -> np.ndarray:
    z = xyz[..., 2]
    valid = (xyz != 0).any(-1)
    if not valid.any():
        return np.zeros(z.shape, np.uint8)
    lo, hi = np.percentile(z[valid], [1, 99])
    g = np.clip((z - lo) / max(hi - lo, 1e-9), 0, 1)
    g = (255 * (1 - g)).astype(np.uint8)   # closer = brighter
    g[~valid] = 0
    return g


def colorize(heat: np.ndarray, vmax: float) -> np.ndarray:
    t = np.clip(heat / max(vmax, 1e-9), 0, 1)
    r = np.clip(3 * t, 0, 1); g = np.clip(3 * t - 1, 0, 1); b = np.clip(np.where(t < 0.5, 1.5 * t, 3 * t - 2), 0, 1)
    rgb = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
    rgb[heat <= 0] = 0
    return rgb


def overlay(depth: np.ndarray, heat: np.ndarray, vmax: float) -> np.ndarray:
    base = np.stack([depth] * 3, -1).astype(np.float32)
    col = colorize(heat, vmax).astype(np.float32)
    a = np.clip(heat / max(vmax, 1e-9), 0, 1)[..., None] * 0.85
    return (base * (1 - a) + col * a).astype(np.uint8)


def inspect(file, category: str):
    if file is None:
        raise gr.Error("Upload an xyz TIFF first.")
    if not category:
        raise gr.Error("Pick a category.")
    try:
        xyz = np.asarray(tifffile.imread(file.name if hasattr(file, "name") else file)).astype(np.float32)
    except Exception as e:  # noqa: BLE001
        raise gr.Error(f"Could not read TIFF: {e}") from e
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise gr.Error(f"Expected HxWx3 organized point cloud, got {xyz.shape}")
    insp = get(category)
    t0 = time.perf_counter()
    try:
        r = insp.inspect(xyz)
    except EmptyScanError as e:
        raise gr.Error(str(e)) from e
    ms = (time.perf_counter() - t0) * 1000
    vmax = (r.threshold or float(r.heatmap.max()) or 1.0) * 1.5
    depth = depth_image(xyz)
    verdict = "FAIL: defect detected" if r.is_defective else "PASS: no defect found"
    summary = (f"## {verdict}\n\n| | |\n|---|---|\n| Anomaly score | {r.image_score:.2f} |\n"
               f"| Threshold | {r.threshold:.2f} |\n| Object points | {r.n_points:,} |\n"
               f"| Latency (CPU) | {ms:.0f} ms |")
    return depth, overlay(depth, r.heatmap, vmax), colorize(r.heatmap, vmax), summary


def build() -> gr.Blocks:
    cats = categories()
    examples = sorted(EXAMPLES.glob("*.tiff")) if EXAMPLES.exists() else []
    with gr.Blocks(title="3D Point Cloud Defect Inspection") as demo:
        gr.Markdown(
            "# 3D Point Cloud Defect Inspection\n"
            "Unsupervised anomaly detection for industrial 3D scans. The model has seen **only defect-free "
            "parts**; anything geometrically unusual lights up. Upload an organized xyz TIFF "
            "(HxWx3 float32, metres, as produced by structured-light scanners) or try an example.\n\n"
            "Method: background removal, FPFH descriptors, nearest-neighbour distance to a coreset memory bank "
            "([Back to the Feature](https://arxiv.org/abs/2303.13194) baseline). "
            "[Code on GitHub](https://github.com/Abdullah-Nasir-Chowdhury/pointcloud-inspection-api)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                f = gr.File(label="xyz scan (.tiff)", file_types=[".tiff", ".tif"])
                c = gr.Dropdown(cats, value=cats[0] if cats else None, label="Part category")
                btn = gr.Button("Inspect", variant="primary")
                out_md = gr.Markdown()
            with gr.Column(scale=2):
                with gr.Row():
                    d = gr.Image(label="Depth", height=300)
                    o = gr.Image(label="Anomaly overlay", height=300)
                    h = gr.Image(label="Heatmap", height=300)
        btn.click(inspect, [f, c], [d, o, h, out_md])
        if examples:
            gr.Examples([[str(p), p.stem.split("__")[0]] for p in examples], inputs=[f, c],
                        outputs=[d, o, h, out_md], fn=inspect, cache_examples=False,
                        label="Examples (MVTec 3D-AD, CC BY-NC-SA 4.0)")
        gr.Markdown("Example scans are from the [MVTec 3D-AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad) "
                    "(Bergmann et al. 2022), CC BY-NC-SA 4.0, used here for non-commercial demonstration.")
    return demo


if __name__ == "__main__":
    build().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
