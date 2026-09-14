"""FastAPI service: POST an organized xyz scan, get an anomaly score, pass/fail and heatmap.

Run:  uvicorn pcinspect.api.app:app --host 0.0.0.0 --port 8000
Env:  PCINSPECT_MODELS=path/to/models   (one sub-folder per category, each with bank.npz + meta.json)
      PCINSPECT_DEMO=path/to/demo       (optional; if gradio is installed the UI is mounted at /demo)
"""
from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path

import numpy as np
import tifffile
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse, Response
from PIL import Image
from pydantic import BaseModel

from ..inspector import InspectionResult, Inspector
from ..preprocess import EmptyScanError

MODELS_DIR = Path(os.environ.get("PCINSPECT_MODELS", "models"))
DEMO_DIR = Path(os.environ.get("PCINSPECT_DEMO", Path(__file__).resolve().parents[2] / "demo"))

app = FastAPI(
    title="Point Cloud Defect Inspection API",
    description="Unsupervised 3D anomaly detection for industrial scans (FPFH + memory bank).",
    version="0.1.0",
)
_inspectors: dict[str, Inspector] = {}


class InspectResponse(BaseModel):
    category: str
    image_score: float
    threshold: float | None
    is_defective: bool | None
    n_points: int
    latency_ms: float
    heatmap_png_base64: str | None = None


def available_categories() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    return sorted(p.name for p in MODELS_DIR.iterdir() if (p / "bank.npz").exists())


def get_inspector(category: str) -> Inspector:
    if category not in _inspectors:
        path = MODELS_DIR / category
        if not (path / "bank.npz").exists():
            raise HTTPException(404, f"unknown category '{category}'; available: {available_categories()}")
        _inspectors[category] = Inspector.load(path)
    return _inspectors[category]


def read_scan(upload: UploadFile, data: bytes) -> np.ndarray:
    name = (upload.filename or "").lower()
    try:
        if name.endswith((".tiff", ".tif")):
            xyz = tifffile.imread(io.BytesIO(data))
        elif name.endswith(".npy"):
            xyz = np.load(io.BytesIO(data))
        else:
            raise HTTPException(415, "upload an organized xyz scan as .tiff (HxWx3 float32) or .npy")
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"could not decode scan: {e}") from e
    xyz = np.asarray(xyz)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise HTTPException(400, f"expected HxWx3 scan, got shape {list(xyz.shape)}")
    return xyz.astype(np.float32)


def heatmap_png(result: InspectionResult, vmax: float | None = None) -> bytes:
    """Colourise the heatmap (matplotlib-free 'inferno'-like ramp) and return PNG bytes."""
    h = result.heatmap
    vmax = vmax or (result.threshold * 1.5 if result.threshold else float(h.max()) or 1.0)
    t = np.clip(h / max(vmax, 1e-9), 0, 1)
    # Simple black -> purple -> orange -> yellow ramp.
    r = np.clip(3 * t, 0, 1)
    g = np.clip(3 * t - 1, 0, 1)
    b = np.clip(np.where(t < 0.5, 1.5 * t, 3 * t - 2), 0, 1)
    rgb = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
    rgb[h <= 0] = 0
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "categories": available_categories()}


@app.get("/categories")
def categories() -> dict:
    out = {}
    for c in available_categories():
        insp = get_inspector(c)
        out[c] = {"bank_size": len(insp.bank), "threshold": insp.threshold}
    return out


@app.post("/inspect", response_model=InspectResponse)
async def inspect(
    category: str = Query(..., description="model to use, see /categories"),
    file: UploadFile = File(..., description="organized xyz scan, .tiff or .npy"),
    heatmap: bool = Query(True, description="include the heatmap as base64 PNG"),
) -> InspectResponse:
    insp = get_inspector(category)
    xyz = read_scan(file, await file.read())
    t0 = time.perf_counter()
    try:
        res = insp.inspect(xyz)
    except EmptyScanError as e:
        raise HTTPException(422, str(e)) from e
    ms = (time.perf_counter() - t0) * 1000
    return InspectResponse(
        category=category, latency_ms=round(ms, 1),
        heatmap_png_base64=base64.b64encode(heatmap_png(res)).decode() if heatmap else None,
        **res.to_json(),
    )


@app.post("/inspect/heatmap.png")
async def inspect_heatmap(
    category: str = Query(...),
    file: UploadFile = File(...),
) -> Response:
    """Same as /inspect but returns only the heatmap PNG (handy for curl and browsers)."""
    insp = get_inspector(category)
    xyz = read_scan(file, await file.read())
    try:
        res = insp.inspect(xyz)
    except EmptyScanError as e:
        raise HTTPException(422, str(e)) from e
    return Response(heatmap_png(res), media_type="image/png",
                    headers={"X-Image-Score": f"{res.image_score:.6f}",
                             "X-Is-Defective": str(res.is_defective)})


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/demo" if _demo_mounted else "/docs")


def _mount_demo() -> bool:
    """Serve the Gradio UI at /demo from the same process, so one container is API + demo.

    Why: a client-facing demo and a machine-facing API on one URL, one deploy, one set of
    loaded models. Gradio is optional: the API works without it.
    """
    try:
        import gradio as gr  # noqa: F401
    except ImportError:
        return False
    if not (DEMO_DIR / "app.py").exists():
        return False
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("pcinspect_demo_app", DEMO_DIR / "app.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pcinspect_demo_app"] = mod
    spec.loader.exec_module(mod)
    gr.mount_gradio_app(app, mod.build(), path="/demo")
    return True


_demo_mounted = _mount_demo()
