import numpy as np
import pytest

from pcinspect.inspector import Inspector, InspectorConfig
from pcinspect.preprocess import choose_voxel_size, PreprocessConfig
from pcinspect.models.memory_bank import MemoryBank, greedy_coreset
from pcinspect.preprocess import EmptyScanError, organized_to_points, preprocess
from tests.synthetic import make_scan


def test_organized_to_points_drops_zeros():
    xyz = np.zeros((4, 4, 3), np.float32)
    xyz[1, 2] = (0.1, 0.2, 0.3)
    pts, idx = organized_to_points(xyz)
    assert pts.shape == (1, 3) and idx.tolist() == [6]


def test_preprocess_removes_table_keeps_object():
    xyz, _ = make_scan()
    pre = preprocess(xyz)
    assert pre.plane is not None
    # Every kept point should be above the table (z < z_table - thresh).
    assert pre.points[:, 2].max() < 0.5 - 0.003
    assert len(pre.down_points) > 100
    assert pre.down_to_full.max() < len(pre.down_points)
    assert pre.shape == (160, 160)


def test_preprocess_rejects_empty():
    with pytest.raises(EmptyScanError):
        preprocess(np.zeros((32, 32, 3), np.float32))


def test_coreset_returns_unique_indices():
    x = np.random.default_rng(0).random((500, 8)).astype(np.float32)
    idx = greedy_coreset(x, 50)
    assert len(idx) == 50 and len(set(idx.tolist())) == 50


def test_memory_bank_roundtrip(tmp_path):
    bank = MemoryBank.fit([np.random.default_rng(0).random((100, 33)).astype(np.float32)])
    bank.save(tmp_path / "b.npz")
    b2 = MemoryBank.load(tmp_path / "b.npz")
    assert len(b2) == 100
    assert np.allclose(b2.score(bank.features[:5]), 0.0, atol=1e-4)


def test_choose_voxel_size_hits_target():
    scans = [make_scan(seed=s)[0] for s in range(3)]
    cfg = PreprocessConfig()
    v = choose_voxel_size(scans, target_points=2000, cfg=cfg)
    assert 0.0005 <= v <= 0.005
    from pcinspect.preprocess import preprocess
    n = len(preprocess(scans[0], PreprocessConfig(voxel_size=v)).down_points)
    assert 1000 < n < 4000, n   # within a factor of two of the target


def test_dent_scores_higher_than_good(tmp_path):
    # Fixed 2 mm voxel: the synthetic sensor pitch is 0.6 mm, so the adaptive voxel would
    # land near the pixel pitch and make normals noisy. This test is about the mechanics.
    cfg = InspectorConfig(target_points=None)
    train = [make_scan(seed=s)[0] for s in range(4)]
    insp = Inspector.fit(train, cfg)
    insp.calibrate([make_scan(seed=s)[0] for s in range(10, 18)])
    good = insp.inspect(make_scan(seed=99)[0])
    bad_xyz, gt = make_scan(dent=True, seed=100)
    bad = insp.inspect(bad_xyz)
    assert bad.image_score > good.image_score
    assert bad.is_defective is True
    assert good.is_defective is False
    # Heatmap should be hotter inside the dent than on the rest of the object.
    obj = bad.heatmap > 0
    assert bad.heatmap[gt].mean() > bad.heatmap[obj & ~gt].mean()
    # Save / load keeps threshold and gives the same score.
    insp.save(tmp_path / "model")
    insp2 = Inspector.load(tmp_path / "model")
    assert insp2.threshold == insp.threshold
    assert abs(insp2.inspect(bad_xyz).image_score - bad.image_score) < 1e-3 * bad.image_score
