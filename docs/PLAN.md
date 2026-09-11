# Project 1: 3D Point Cloud Defect Inspection API

Portfolio project 1 of 10 (see E:\Freelancing\upwork-portfolio-checklist.md). Budget ~40 h.
Goal: upload an industrial 3D scan, get back a per-point anomaly heatmap and a calibrated
pass/fail decision. Public benchmark: MVTec 3D-AD (10 categories, defect-free training only).

## Method (v1, no training, CPU-friendly)

Unsupervised anomaly detection with a geometric memory bank, following
"Back to the Feature" (Horwitz & Hoshen, 2023): handcrafted FPFH descriptors beat most
learned 3D features on MVTec 3D-AD.

1. Preprocess (Open3D): load organized xyz tiff -> drop zero points -> RANSAC plane fit to
   remove the background table -> voxel downsample -> estimate normals.
2. Describe: FPFH (33-d) per downsampled point. Optional: fuse with RGB PatchCore features
   from a frozen WideResNet50 for a "3D+RGB" variant.
3. Memory bank: FPFH features from all defect-free training scans of a category, reduced
   with greedy coreset subsampling (~10%). Store as a .npz per category.
4. Score: for each test point, distance to nearest bank feature (sklearn NearestNeighbors
   or FAISS). Point scores -> Gaussian-smoothed -> reprojected onto the HxW image grid via
   the organized-cloud index (pixel heatmap). Image score = mean of top-k point scores.
5. Threshold: calibrated on the validation split (defect-free) as the 99th percentile of
   image scores; the API returns pass/fail plus the raw score and the threshold used.

Why this and not the ACCV method directly: the paper is under review until acceptance, so
the public project uses a published baseline, and the ACCV ideas can slot in later as an
"experimental" scorer behind a flag.

## Evaluation protocol

Standard MVTec 3D-AD metrics, reported per category and as the mean over 10:
- Image-level AUROC (detection)
- Pixel-level AUROC and AUPRO at 30% FPR integration limit (localisation)
- Latency per scan, CPU (i7 class) and GPU (RTX 3070), plus peak RAM
Reference numbers to beat or match (BTF, FPFH only): I-AUROC 0.865, AUPRO 0.924.

## Deliverables (per the checklist standards)

- [x] `pcinspect` package: data loader, preprocessing, FPFH features, memory-bank scorer
- [x] `scripts/build_bank.py`, `scripts/evaluate.py`, `scripts/benchmark.py`
- [x] FastAPI service: `POST /inspect` (tiff/ply upload -> JSON score + heatmap PNG),
      `GET /health`, `GET /categories`
- [x] Dockerfile (python:3.11-slim, CPU only); build blocked until Docker Desktop is installed
- [~] Hugging Face Space: assembled and tested locally (`scripts/build_space.py`); upload blocked, HF requires PRO for Gradio Spaces (402 on 2026-09-11)
- [ ] README: client problem, results table, run steps, 60-90 s GIF
- [x] Benchmark table and one client-facing paragraph for proposals (`docs/CLIENT_PARAGRAPH.md`)
- [x] Tests: loader, preprocessing invariants, API round trip on a synthetic sample

## Milestones

| # | Milestone | Hours | Done when |
|---|---|---|---|
| M1 | Data + preprocessing + FPFH on one category (bagel) | 6 | Heatmap on a real defect looks right |
| M2 | Memory bank + scoring + metrics on all 10 categories | 10 | Mean I-AUROC within 0.02 of BTF |
| M3 | FastAPI service + tests + Dockerfile | 8 | `docker compose up` then curl returns a heatmap |
| M4 | HF Space demo + benchmark table | 8 | Public URL works from a phone |
| M5 | README, GIF, client paragraph, repo public | 4 | Checklist standards all ticked |
| — | Buffer (RGB fusion, ACCV scorer flag) | 4 | Optional |

## Environment

- Windows 11, RTX 3070 8 GB, Python 3.11 venv at `.venv` (Open3D 0.19 has no 3.14 wheels)
- Docker not installed yet; WSL2 Ubuntu is present, so Docker Desktop or Docker-in-WSL works
- Dataset: ~11.6 GB archive, ~14 GB extracted, under `data/mvtec_3d/` (git-ignored)
