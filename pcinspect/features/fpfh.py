"""Fast Point Feature Histograms (Rusu et al. 2009) via Open3D. 33-d per point."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class FPFHConfig:
    radius_mult: float = 5.0   # feature radius = mult * voxel_size
    max_nn: int = 100


def compute_fpfh(points: np.ndarray, normals: np.ndarray, voxel_size: float,
                 cfg: FPFHConfig = FPFHConfig()) -> np.ndarray:
    """Return (M, 33) float32 FPFH descriptors for the given points/normals."""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.normals = o3d.utility.Vector3dVector(normals)
    feat = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * cfg.radius_mult, max_nn=cfg.max_nn)
    )
    return np.asarray(feat.data).T.astype(np.float32)   # Open3D returns (33, M)
