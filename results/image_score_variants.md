# Image score variants (offline, same banks, I-AUROC)

Tested after tire came out below chance. Per-point scores are nearest-neighbour distances;
the image score aggregates them. Smoothing uses normalised convolution over the object mask.

| Category | top-1% mean (default) | max | smoothed max (s=4) | smoothed max (s=8) | smoothed top-1% (s=8) | median of top-5% | mean |
|---|---|---|---|---|---|---|---|
| tire | 0.396 | 0.414 | 0.392 | 0.375 | 0.381 | 0.483 | 0.627 |
| cable_gland | 0.680 | 0.602 | 0.632 | 0.687 | 0.719 | 0.722 | 0.690 |
| bagel | 0.974 | 0.886 | 0.946 | 0.954 | 0.970 | 0.941 | 0.847 |

Conclusion: no peak-based aggregate fixes tire; only the whole-scan mean exceeds chance, so tire
defects shift the distribution slightly without producing the strongest peaks. Kept top-1% mean
(best on bagel, close to best elsewhere). Tire is reported as a known failure case of FPFH
features on finely grooved black rubber; a learned or RGB-fused feature is the next step.
