# Learning notes

You asked to learn along the way. This file lists every concept the code uses, why it was
chosen, and what to read. Work through it top to bottom; each section names the file to open.

## 1. The data: organized point clouds (`pcinspect/data/mvtec3d.py`)

MVTec 3D-AD stores each scan as an HxWx3 float32 TIFF: for every camera pixel, the (x, y, z)
of the surface point it saw, in metres, and (0, 0, 0) where the sensor got no return.
This is an *organized* cloud: it has an image grid, so we can paint per-point scores back to
pixels and compare with the ground-truth mask. Structured-light sensors (Zivid, Photoneo,
Keyence) output exactly this, which is why clients can drop their own scans into the API.

Try: load one scan, print `xyz.shape`, count zero pixels, and plot `xyz[..., 2]` as an image.

## 2. Preprocessing (`pcinspect/preprocess.py`)

- **Plane removal with RANSAC.** The object sits on a flat table. RANSAC picks 3 random points,
  fits a plane, counts inliers, repeats, keeps the best. We then keep only points on the camera
  side of that plane. Read: Fischler & Bolles 1981 (the original RANSAC paper is 6 pages).
  Question to answer yourself: why did I flip the plane so the camera has positive distance?
- **Voxel downsampling.** Snap points to a 2 mm grid and average each cell. Makes point density
  uniform, so descriptors are comparable between scans and the memory bank stays small.
- **Normal estimation.** PCA on each point's neighbours; the smallest eigenvector is the
  normal. `orient_normals_towards_camera_location` fixes the sign ambiguity.

Try: change `voxel_size` to 0.001 and 0.004 and watch bank size and latency change.

## 3. FPFH descriptors (`pcinspect/features/fpfh.py`)

Fast Point Feature Histograms (Rusu 2009) describe local surface shape around a point as a
33-bin histogram of angles between normals of neighbouring pairs. Rotation invariant, cheap,
and the "Back to the Feature" paper (Horwitz & Hoshen 2023) showed they beat most learned 3D
features for anomaly detection. Read that paper first; it is short and the whole method here
is its 3D branch.

## 4. Memory bank anomaly detection (`pcinspect/models/memory_bank.py`)

PatchCore idea (Roth et al. 2022): store features from *normal* data; at test time, a feature
far from every stored feature is anomalous. Score = nearest-neighbour distance. No training,
no labels, one hyperparameter (bank size).

- **FAISS** does brute-force nearest neighbour fast on CPU. `IndexFlatL2` is exact.
- **Greedy coreset** (k-center) picks a subset that covers the feature space evenly, so we can
  shrink the bank 10x with little loss. It is optional here; random per-scan capping is the
  default because it is fast. Try turning `coreset_fraction=0.1` on and compare AUROC.

## 5. From point scores to a decision (`pcinspect/inspector.py`)

- Scores live on downsampled points; `down_to_full` maps them to every full-res point, which
  maps to a pixel through `pix_idx`.
- **Normalised convolution** for smoothing: blur the heatmap *and* the object mask, divide.
  Plain Gaussian blur would drag object scores toward zero near the background.
- **Image score = mean of the top 1% point scores**, not the max: the max is one noisy point.
- **Threshold = 99th percentile of validation scores.** Validation scans are all defect-free,
  so this is "allow ~1% false alarms". A client would tune this to their cost of a missed defect.

## 6. Metrics (`pcinspect/metrics.py`)

- **Image AUROC**: can the score rank defective scans above good ones? Threshold-free.
- **Pixel AUROC**: same, per pixel. Inflated by the huge number of easy background pixels.
- **AUPRO**: per-region overlap, integrated up to 30% false positive rate. Treats a small
  defect and a large defect equally, which pixel AUROC does not. This is the MVTec 3D-AD
  localisation metric everyone reports. Read Bergmann et al. 2021 section on PRO.

## 7. Serving (`pcinspect/api/app.py`)

FastAPI turns typed Python functions into an HTTP API with automatic docs at `/docs`.
Models are lazy-loaded per category and cached in a dict. Try: start the server, open
`http://localhost:8000/docs`, and upload a TIFF from the browser.

## 8. Testing without data (`tests/`)

`tests/synthetic.py` builds a fake sensor scan (table + dome + optional dent). This let the
whole pipeline be verified before the 14 GB dataset arrived. Pattern worth copying: always have
a tiny synthetic fixture so tests run in CI in seconds.

## Reading order

1. Horwitz & Hoshen, "Back to the Feature", 2023 (arXiv 2303.13194)
2. Roth et al., "Towards Total Recall in Industrial Anomaly Detection" (PatchCore), 2022
3. Bergmann et al., "The MVTec 3D-AD Dataset", 2022 (arXiv 2112.09045)
4. Open3D tutorials: point cloud basics, plane segmentation, FPFH/global registration
5. FastAPI tutorial, first 6 sections

## Your tasks (do these yourself, then we review)

- [ ] Reproduce the synthetic test by hand in a notebook and plot the heatmap
- [ ] Sweep `voxel_size` in {1, 2, 4} mm on the bagel category and record I-AUROC and latency
- [ ] Write `scripts/visualize.py`: RGB | depth | heatmap | ground truth side by side for one scan
- [ ] Explain in one paragraph (README) why nearest-neighbour distance is an anomaly score
