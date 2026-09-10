"""MVTec-style anomaly metrics: image AUROC, pixel AUROC, AUPRO@30% FPR."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import label
from sklearn.metrics import roc_auc_score


def image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels.astype(int), scores))


def pixel_auroc(gts: list[np.ndarray], maps: list[np.ndarray], max_pixels: int = 2_000_000,
                seed: int = 0) -> float:
    """Pixel AUROC over all test images (subsampled for memory)."""
    y = np.concatenate([g.ravel() for g in gts]).astype(np.int8)
    s = np.concatenate([m.ravel() for m in maps]).astype(np.float32)
    if len(y) > max_pixels:
        idx = np.random.default_rng(seed).choice(len(y), max_pixels, replace=False)
        y, s = y[idx], s[idx]
    return float(roc_auc_score(y, s))


def aupro(gts: list[np.ndarray], maps: list[np.ndarray], fpr_limit: float = 0.3,
          n_thresholds: int = 200) -> float:
    """Area under the per-region-overlap curve up to `fpr_limit`, normalised to [0, 1].

    Bergmann et al. 2021 definition: at each threshold, PRO is the mean over every
    ground-truth connected component of the fraction of that component covered by
    the prediction; FPR is computed over all defect-free pixels.
    """
    comps: list[tuple[np.ndarray, np.ndarray, int]] = []   # (map, comp_label_img, n_comps)
    neg_pixels = 0
    all_scores = []
    for g, m in zip(gts, maps):
        lab, n = label(g)
        comps.append((m, lab, n))
        neg_pixels += int((~g).sum())
        all_scores.append(m.ravel())
    scores = np.concatenate(all_scores)
    ths = np.quantile(scores[scores > 0], np.linspace(0, 1, n_thresholds)) if (scores > 0).any() \
        else np.linspace(0, 1, n_thresholds)
    ths = np.unique(ths)[::-1]   # high -> low so FPR increases

    fprs, pros = [0.0], [0.0]
    for t in ths:
        fp = 0
        overlaps = []
        for m, lab, n in comps:
            pred = m >= t
            fp += int((pred & (lab == 0)).sum())
            if n:
                sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
                hits = np.bincount(lab.ravel(), weights=pred.ravel(), minlength=n + 1)[1:]
                overlaps.extend(hits / np.maximum(sizes, 1))
        fpr = fp / max(neg_pixels, 1)
        fprs.append(fpr)
        pros.append(float(np.mean(overlaps)) if overlaps else 0.0)
        if fpr >= fpr_limit:
            break
    fprs, pros = np.asarray(fprs), np.asarray(pros)
    if fprs[-1] > fpr_limit:   # interpolate the last point onto the limit
        pros[-1] = np.interp(fpr_limit, fprs[-2:], pros[-2:])
        fprs[-1] = fpr_limit
    return float(np.trapezoid(pros, fprs) / fpr_limit)
