# Client-facing paragraph (for Upwork proposals and profile)

I build 3D inspection systems that find defects without needing labelled examples of them.
For a recent project I took raw structured-light scans of ten different parts (bagels to
cable glands to tires), learned what "good" looks like from defect-free scans only, and
delivered a REST API that returns a per-pixel anomaly heatmap and a pass/fail decision in
under a second on a CPU. On the public MVTec 3D-AD benchmark it detects defects with a mean
image AUROC of 0.84 across categories, above the published baseline for the same
features, and it ships as a Docker image with tests, a benchmark table and a live demo. If you
have a scanner and a bin of good parts, I can have a first model on your data within a week and
tell you honestly which defect types it will and will not catch.

# Shorter variant (Project Catalog)

Unsupervised 3D defect detection from your scanner's point clouds: heatmap + pass/fail via
REST API, trained on good parts only, Dockerised, benchmarked. Delivered with an evaluation
report on your own scans.
