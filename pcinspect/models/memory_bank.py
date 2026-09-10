"""Nearest-neighbour memory bank of normal features (PatchCore-style)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except ImportError:  # pragma: no cover
    from sklearn.neighbors import NearestNeighbors
    _HAS_FAISS = False


def greedy_coreset(x: np.ndarray, n_keep: int, proj_dim: int = 16, seed: int = 0) -> np.ndarray:
    """Greedy k-center coreset on a random projection (Sinha et al. 2020). Returns indices."""
    rng = np.random.default_rng(seed)
    if n_keep >= len(x):
        return np.arange(len(x))
    proj = x @ rng.standard_normal((x.shape[1], proj_dim)).astype(x.dtype)
    chosen = [int(rng.integers(len(x)))]
    min_d = np.linalg.norm(proj - proj[chosen[0]], axis=1)
    for _ in range(n_keep - 1):
        i = int(np.argmax(min_d))
        chosen.append(i)
        min_d = np.minimum(min_d, np.linalg.norm(proj - proj[i], axis=1))
    return np.asarray(chosen)


class MemoryBank:
    def __init__(self, features: np.ndarray):
        self.features = np.ascontiguousarray(features, dtype=np.float32)
        if _HAS_FAISS:
            self._index = faiss.IndexFlatL2(self.features.shape[1])
            self._index.add(self.features)
        else:
            self._index = NearestNeighbors(n_neighbors=1).fit(self.features)

    @classmethod
    def fit(cls, feature_sets: list[np.ndarray], max_per_sample: int | None = 4000,
            coreset_fraction: float | None = None, seed: int = 0) -> "MemoryBank":
        """Build a bank from per-scan feature arrays.

        max_per_sample: random cap per scan (cheap, keeps every scan represented).
        coreset_fraction: if set, additionally reduce the pooled bank by greedy coreset.
        """
        rng = np.random.default_rng(seed)
        pooled = []
        for f in feature_sets:
            if max_per_sample is not None and len(f) > max_per_sample:
                f = f[rng.choice(len(f), max_per_sample, replace=False)]
            pooled.append(f)
        bank = np.concatenate(pooled).astype(np.float32)
        if coreset_fraction is not None and 0 < coreset_fraction < 1:
            bank = bank[greedy_coreset(bank, int(len(bank) * coreset_fraction), seed=seed)]
        return cls(bank)

    def score(self, features: np.ndarray) -> np.ndarray:
        """Euclidean distance from each query feature to its nearest bank entry, (Q,)."""
        q = np.ascontiguousarray(features, dtype=np.float32)
        if _HAS_FAISS:
            d2, _ = self._index.search(q, 1)
            return np.sqrt(np.maximum(d2[:, 0], 0.0))
        d, _ = self._index.kneighbors(q, n_neighbors=1)
        return d[:, 0]

    def save(self, path: Path) -> None:
        np.savez_compressed(path, features=self.features)

    @classmethod
    def load(cls, path: Path) -> "MemoryBank":
        return cls(np.load(path)["features"])

    def __len__(self) -> int:
        return len(self.features)
