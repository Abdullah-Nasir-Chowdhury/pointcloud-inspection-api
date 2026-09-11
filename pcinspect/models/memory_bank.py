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
    """Greedy k-center coreset on a random projection (PatchCore, Roth et al. 2022).

    Picks points one at a time, always the one farthest from everything already picked,
    so the subset covers the feature space evenly. Distances are computed in a `proj_dim`
    random projection (Johnson-Lindenstrauss) to make each step cheap. Returns indices.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    if n_keep >= n:
        return np.arange(n)
    proj = np.ascontiguousarray(x @ rng.standard_normal((x.shape[1], proj_dim)).astype(np.float32))
    sq = np.einsum("ij,ij->i", proj, proj)
    chosen = np.empty(n_keep, dtype=np.int64)
    chosen[0] = rng.integers(n)
    # squared distance to the nearest chosen point so far
    min_d = sq - 2 * (proj @ proj[chosen[0]]) + sq[chosen[0]]
    for k in range(1, n_keep):
        i = int(np.argmax(min_d))
        chosen[k] = i
        d = sq - 2 * (proj @ proj[i]) + sq[i]
        np.minimum(min_d, d, out=min_d)
    return chosen


class MemoryBank:
    def __init__(self, features: np.ndarray):
        self.features = np.ascontiguousarray(features, dtype=np.float32)
        if _HAS_FAISS:
            self._index = faiss.IndexFlatL2(self.features.shape[1])
            self._index.add(self.features)
        else:
            self._index = NearestNeighbors(n_neighbors=1).fit(self.features)

    @classmethod
    def fit(cls, feature_sets: list[np.ndarray], max_per_sample: int | None = 1000,
            coreset_size: int | None = 40000, seed: int = 0) -> "MemoryBank":
        """Build a bank from per-scan feature arrays.

        max_per_sample: random cap per scan (cheap, keeps every scan represented).
        coreset_size: if set, reduce the pooled bank to this many entries by greedy coreset.
            Search time and model size scale linearly with bank size; 40k keeps a scan
            under ~1 s on CPU.
        """
        rng = np.random.default_rng(seed)
        pooled = []
        for f in feature_sets:
            if max_per_sample is not None and len(f) > max_per_sample:
                f = f[rng.choice(len(f), max_per_sample, replace=False)]
            pooled.append(f)
        bank = np.concatenate(pooled).astype(np.float32)
        if coreset_size is not None and coreset_size < len(bank):
            bank = bank[greedy_coreset(bank, coreset_size, seed=seed)]
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
