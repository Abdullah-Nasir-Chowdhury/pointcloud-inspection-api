"""Turn an organized xyz scan into a clean object point cloud.

Steps: drop no-return pixels -> RANSAC plane fit to find the background table ->
keep points in front of the plane -> voxel downsample -> estimate normals.
Every retained full-resolution point keeps its pixel index so scores can be painted
back onto the HxW image grid.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


@dataclass
class Preprocessed:
    shape: tuple[int, int]            # (H, W) of the source scan
    pix_idx: np.ndarray               # flat pixel index of each full-res object point, (N,)
    points: np.ndarray                # full-res object points, (N, 3)
    down_points: np.ndarray           # downsampled points, (M, 3)
    down_normals: np.ndarray          # (M, 3)
    down_to_full: np.ndarray          # for each full-res point, index of its nearest down point, (N,)
    plane: np.ndarray | None          # (a, b, c, d) with normal facing the camera, or None if skipped


@dataclass(frozen=True)
class PreprocessConfig:
    plane_dist_thresh: float = 0.004  # m: points closer than this to the plane are background
    plane_ransac_iters: int = 1000
    min_object_fraction: float = 0.02 # if the plane removal keeps less than this, skip it
    voxel_size: float = 0.002         # m
    normal_radius_mult: float = 2.5   # normal radius = mult * voxel_size
    normal_max_nn: int = 30
    min_points: int = 200             # below this the scan is rejected as empty
    seed: int = 0                     # RANSAC seed


class EmptyScanError(ValueError):
    pass


def organized_to_points(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(H,W,3) -> (valid points (N,3), flat pixel indices (N,))."""
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError(f"expected HxWx3, got {xyz.shape}")
    flat = xyz.reshape(-1, 3)
    valid = np.any(flat != 0, axis=1) & np.all(np.isfinite(flat), axis=1)
    idx = np.flatnonzero(valid)
    return flat[idx].astype(np.float64), idx


def remove_background_plane(points: np.ndarray, cfg: PreprocessConfig) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (mask of object points, plane) using RANSAC on the dominant plane.

    The plane normal is flipped to face the camera at the origin, so the object
    (which sits between camera and table) has positive signed distance.
    """
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if hasattr(o3d.utility, "random"):
        o3d.utility.random.seed(cfg.seed)  # make RANSAC reproducible
    plane, _ = pcd.segment_plane(
        distance_threshold=cfg.plane_dist_thresh, ransac_n=3, num_iterations=cfg.plane_ransac_iters
    )
    plane = np.asarray(plane, dtype=np.float64)
    n, d = plane[:3], plane[3]
    # Signed distance of the camera (origin) is d / |n|; make it positive.
    if d < 0:
        plane = -plane
        n, d = plane[:3], plane[3]
    signed = (points @ n + d) / np.linalg.norm(n)
    keep = signed > cfg.plane_dist_thresh
    if keep.mean() < cfg.min_object_fraction:
        return np.ones(len(points), dtype=bool), None
    return keep, plane


def choose_voxel_size(scans: list[np.ndarray], target_points: int, cfg: PreprocessConfig,
                      probe_voxel: float = 0.001, lo: float = 0.0005, hi: float = 0.005) -> float:
    """Pick a voxel size so a typical object yields about `target_points` downsampled points.

    Why: FPFH neighbourhoods are defined in voxels, so a fixed 2 mm voxel gives a bagel
    4,000 points but a cable gland 1,000. Normalising the point budget per category makes
    small parts as well resolved as large ones. Surface point count scales ~ 1/voxel^2.
    """
    counts = []
    for xyz in scans:
        points, _ = organized_to_points(xyz)
        keep, _ = remove_background_plane(points, cfg)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[keep]))
        counts.append(len(pcd.voxel_down_sample(probe_voxel).points))
    n = float(np.median(counts))
    return float(np.clip(probe_voxel * np.sqrt(n / target_points), lo, hi))


def preprocess(xyz: np.ndarray, cfg: PreprocessConfig = PreprocessConfig()) -> Preprocessed:
    H, W = xyz.shape[:2]
    points, pix_idx = organized_to_points(xyz)
    if len(points) < cfg.min_points:
        raise EmptyScanError(f"only {len(points)} valid points")

    keep, plane = remove_background_plane(points, cfg)
    points, pix_idx = points[keep], pix_idx[keep]
    if len(points) < cfg.min_points:
        raise EmptyScanError(f"only {len(points)} object points after plane removal")

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    down = pcd.voxel_down_sample(cfg.voxel_size)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=cfg.voxel_size * cfg.normal_radius_mult, max_nn=cfg.normal_max_nn
        )
    )
    down.orient_normals_towards_camera_location(np.zeros(3))
    down_points = np.asarray(down.points)
    down_normals = np.asarray(down.normals)

    _, down_to_full = cKDTree(down_points).query(points, k=1)
    return Preprocessed(
        shape=(H, W),
        pix_idx=pix_idx,
        points=points,
        down_points=down_points,
        down_normals=down_normals,
        down_to_full=down_to_full.astype(np.int64),
        plane=plane,
    )
