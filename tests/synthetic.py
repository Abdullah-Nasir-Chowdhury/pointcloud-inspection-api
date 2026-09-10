"""Synthetic organized scans: a flat table with a dome-shaped object, optional dent."""
from __future__ import annotations

import numpy as np


def make_scan(H: int = 160, W: int = 160, z_table: float = 0.5, radius: float = 0.03,
              height: float = 0.02, dent: bool = False, dent_depth: float = 0.006,
              noise: float = 0.0001, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (xyz HxWx3 in metres, gt mask HxW of the dent)."""
    rng = np.random.default_rng(seed)
    pitch = 0.0006  # 0.6 mm per pixel -> ~10 cm field of view
    u = (np.arange(W) - W / 2) * pitch
    v = (np.arange(H) - H / 2) * pitch
    X, Y = np.meshgrid(u, v)
    r2 = X**2 + Y**2
    # Dome: paraboloid sitting on the table, facing the camera at z=0.
    dome = np.where(r2 < radius**2, height * (1 - r2 / radius**2), 0.0)
    gt = np.zeros((H, W), dtype=bool)
    if dent:
        cx, cy, dr = 0.01, 0.0, 0.006
        d2 = (X - cx) ** 2 + (Y - cy) ** 2
        bump = np.where(d2 < dr**2, dent_depth * (1 - d2 / dr**2), 0.0)
        dome = np.where(dome > 0, dome - bump, dome)
        gt = (d2 < dr**2) & (r2 < radius**2)
    Z = z_table - dome + rng.normal(0, noise, (H, W))
    xyz = np.stack([X, Y, Z], axis=-1).astype(np.float32)
    # Sprinkle no-return pixels like a real sensor.
    drop = rng.random((H, W)) < 0.02
    xyz[drop] = 0
    return xyz, gt
