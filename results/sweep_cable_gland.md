# Sweep: cable_gland (point budget), corrected AUPRO

| Target points | Voxel (mm) | I-AUROC | P-AUROC | AUPRO@30% | Acc@thr | Latency (ms) |
|---|---|---|---|---|---|---|
| fixed 2 mm | 2.00 | 0.633 | 0.899 | 0.736 | 0.269 | 134 |
| 3000 | 1.00 | 0.644 | 0.902 | 0.771 | 0.287 | 378 |
| 6000 | 0.71 | 0.627 | 0.903 | 0.779 | 0.324 | 938 |
| 12000 | 0.50 | 0.545 | 0.907 | 0.794 | 0.194 | 1273 |

Chosen default: 3000 points (best I-AUROC, +0.035 AUPRO over fixed voxel, <0.4 s/scan).
