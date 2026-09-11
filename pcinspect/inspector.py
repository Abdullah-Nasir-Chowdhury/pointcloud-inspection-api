"""End-to-end scorer: organized xyz scan -> pixel heatmap + image score + pass/fail."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from .features.fpfh import FPFHConfig, compute_fpfh
from .models.memory_bank import MemoryBank
from .preprocess import PreprocessConfig, Preprocessed, preprocess


@dataclass(frozen=True)
class InspectorConfig:
    pre: PreprocessConfig = PreprocessConfig()
    fpfh: FPFHConfig = FPFHConfig()
    heatmap_sigma: float = 4.0     # px, Gaussian smoothing of the reprojected map
    topk_fraction: float = 0.01    # image score = mean of the top 1% point scores
    max_per_sample: int | None = 1000
    coreset_size: int | None = 40000


@dataclass
class InspectionResult:
    image_score: float
    threshold: float | None
    is_defective: bool | None
    heatmap: np.ndarray            # (H, W) float32, 0 on background
    point_scores: np.ndarray       # per full-res object point
    n_points: int

    def to_json(self) -> dict:
        return {
            "image_score": float(self.image_score),
            "threshold": None if self.threshold is None else float(self.threshold),
            "is_defective": self.is_defective,
            "n_points": int(self.n_points),
        }


def extract_features(xyz: np.ndarray, cfg: InspectorConfig) -> tuple[Preprocessed, np.ndarray]:
    pre = preprocess(xyz, cfg.pre)
    feats = compute_fpfh(pre.down_points, pre.down_normals, cfg.pre.voxel_size, cfg.fpfh)
    return pre, feats


class Inspector:
    def __init__(self, bank: MemoryBank, cfg: InspectorConfig = InspectorConfig(),
                 threshold: float | None = None):
        self.bank = bank
        self.cfg = cfg
        self.threshold = threshold

    # ---- training -------------------------------------------------------------------
    @classmethod
    def fit(cls, train_scans: list[np.ndarray], cfg: InspectorConfig = InspectorConfig()) -> "Inspector":
        feats = [extract_features(x, cfg)[1] for x in train_scans]
        bank = MemoryBank.fit(feats, max_per_sample=cfg.max_per_sample,
                              coreset_size=cfg.coreset_size)
        return cls(bank, cfg)

    def calibrate(self, val_scans: list[np.ndarray], percentile: float = 99.0, margin: float = 1.0) -> float:
        """Set the pass/fail threshold from defect-free validation scans."""
        scores = np.array([self.inspect(x).image_score for x in val_scans])
        self.threshold = float(np.percentile(scores, percentile) * margin)
        return self.threshold

    # ---- inference ------------------------------------------------------------------
    def inspect(self, xyz: np.ndarray) -> InspectionResult:
        pre, feats = extract_features(xyz, self.cfg)
        down_scores = self.bank.score(feats)
        point_scores = down_scores[pre.down_to_full]

        H, W = pre.shape
        heat = np.zeros(H * W, dtype=np.float32)
        heat[pre.pix_idx] = point_scores
        heat = heat.reshape(H, W)
        mask = np.zeros(H * W, dtype=np.float32)
        mask[pre.pix_idx] = 1.0
        mask = mask.reshape(H, W)
        if self.cfg.heatmap_sigma > 0:
            # Normalised convolution so background zeros do not dilute object scores.
            num = gaussian_filter(heat, self.cfg.heatmap_sigma)
            den = gaussian_filter(mask, self.cfg.heatmap_sigma)
            heat = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0) * (mask > 0)

        k = max(1, int(len(point_scores) * self.cfg.topk_fraction))
        image_score = float(np.sort(point_scores)[-k:].mean())
        is_def = None if self.threshold is None else bool(image_score > self.threshold)
        return InspectionResult(image_score, self.threshold, is_def, heat.astype(np.float32),
                                point_scores.astype(np.float32), len(point_scores))

    # ---- persistence ----------------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.bank.save(directory / "bank.npz")
        meta = {"threshold": self.threshold, "config": asdict(self.cfg)}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path) -> "Inspector":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        c = meta["config"]
        cfg = InspectorConfig(pre=PreprocessConfig(**c["pre"]), fpfh=FPFHConfig(**c["fpfh"]),
                              **{k: v for k, v in c.items() if k not in ("pre", "fpfh")})
        return cls(MemoryBank.load(directory / "bank.npz"), cfg, meta["threshold"])
