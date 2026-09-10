# Point Cloud Defect Inspection API

> Status: work in progress (started 2026-09-11). Pipeline and API pass synthetic tests; MVTec 3D-AD benchmark pending.

Upload a 3D scan of a manufactured part, get back an anomaly heatmap and a calibrated
pass/fail decision. No labelled defects needed: the model learns from defect-free scans only.

## Client problem

Manufacturing QA teams have 3D scanners but no defect labels. Supervised detectors need
hundreds of annotated defects per part type; this service needs 50-300 good scans and
flags anything that deviates geometrically (dents, holes, cracks, contamination, bends).

## How it works

1. Background plane removal and voxel downsampling (Open3D)
2. FPFH geometric descriptors per point
3. Nearest-neighbour distance to a memory bank of defect-free features (FAISS)
4. Scores painted back onto the sensor image grid as a heatmap; image score = top-1% mean
5. Threshold calibrated on defect-free validation scans

Method follows "Back to the Feature" (Horwitz and Hoshen, CVPR-W 2023).

## Results on MVTec 3D-AD

_Pending: table produced by `scripts/evaluate.py`._

## Run

```bash
python -m venv .venv && .venv/Scripts/pip install -e .[dev] faiss-cpu
python scripts/download_mvtec3d.py --categories bagel      # or no flag for all 13 GB
python scripts/build_bank.py --categories bagel
python scripts/evaluate.py --categories bagel
uvicorn pcinspect.api.app:app --port 8000
curl -F "file=@scan.tiff" "localhost:8000/inspect?category=bagel"
```

Docker: `docker build -t pcinspect . && docker run -p 8000:8000 pcinspect`

## Tests

```bash
pytest -q
```

## Data license

MVTec 3D-AD is CC BY-NC-SA 4.0 (non-commercial). Nothing from the dataset is redistributed here.
