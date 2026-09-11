"""Assemble (and optionally upload) the Hugging Face Space for the Gradio demo.

    python scripts/build_space.py                    # assemble into runs/space/
    python scripts/build_space.py --upload user/name --private

The Space is a plain Gradio app: app.py + requirements.txt + the pcinspect package +
models/<category>/ (5 MB each) + examples/. Everything is copied, nothing symlinked,
so the folder is self-contained.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "space"

README = """---
title: 3D Point Cloud Defect Inspection
emoji: 🔍
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: {gradio_version}
app_file: app.py
pinned: false
license: mit
short_description: Unsupervised defect detection on industrial 3D scans
---

# 3D Point Cloud Defect Inspection

Upload an organized xyz scan (HxWx3 float32 TIFF, metres) of a manufactured part and get an
anomaly heatmap plus a pass/fail decision. The model has only ever seen defect-free parts.

Method: background plane removal, FPFH descriptors, nearest-neighbour distance to a greedy
coreset memory bank (Back to the Feature baseline). Runs on CPU in about a second per scan.

Code, benchmark and API: https://github.com/Abdullah-Nasir-Chowdhury/pointcloud-inspection-api

Example scans are from MVTec 3D-AD (CC BY-NC-SA 4.0), used for non-commercial demonstration.
"""

REQUIREMENTS = """open3d==0.19.0
numpy
scipy
scikit-learn
faiss-cpu
tifffile
pillow
gradio=={gradio_version}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", metavar="REPO_ID", help="e.g. AbdullahNasir/pointcloud-defect-inspection")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    args = ap.parse_args()

    import gradio
    gv = gradio.__version__

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    shutil.copytree(ROOT / "pcinspect", OUT / "pcinspect",
                    ignore=shutil.ignore_patterns("__pycache__", "api"))
    shutil.copy(ROOT / "demo" / "app.py", OUT / "app.py")
    shutil.copytree(ROOT / "demo" / "examples", OUT / "examples")
    for cat in sorted(p for p in args.models.iterdir() if (p / "bank.npz").exists()):
        shutil.copytree(cat, OUT / "models" / cat.name)
    (OUT / "README.md").write_text(README.format(gradio_version=gv))
    (OUT / "requirements.txt").write_text(REQUIREMENTS.format(gradio_version=gv))
    # app.py resolves models relative to its own folder and imports pcinspect from its parent;
    # in the Space both live next to app.py, so point it there.
    app = (OUT / "app.py").read_text()
    app = app.replace('sys.path.insert(0, str(HERE.parent))', 'sys.path.insert(0, str(HERE))')
    app = app.replace('HERE.parent / "models"', 'HERE / "models"')
    (OUT / "app.py").write_text(app)
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"assembled {OUT} ({size:.0f} MB)")

    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.upload, repo_type="space", space_sdk="gradio", private=args.private, exist_ok=True)
        api.upload_folder(folder_path=str(OUT), repo_id=args.upload, repo_type="space",
                          commit_message="Deploy demo")
        print(f"uploaded: https://huggingface.co/spaces/{args.upload}")


if __name__ == "__main__":
    main()
