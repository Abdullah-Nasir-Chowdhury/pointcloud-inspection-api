import numpy as np

from pcinspect.metrics import aupro, pixel_auroc


def _small_object(obj_frac_side=0.3, seed=0):
    """100x100 image, object occupies a central square, defect is a blob inside it.
    Background (outside object) scores exactly 0 like the real heatmaps."""
    rng = np.random.default_rng(seed)
    H = W = 100
    s = int(H * obj_frac_side)
    o0 = (H - s) // 2
    obj = np.zeros((H, W), bool); obj[o0:o0 + s, o0:o0 + s] = True
    gt = np.zeros((H, W), bool); gt[o0 + 2:o0 + 6, o0 + 2:o0 + 6] = True
    m = rng.random((H, W)).astype(np.float32) * 0.1 + 0.01
    m[~obj] = 0
    return obj, gt, m


def test_aupro_perfect_map_is_one_regardless_of_object_size():
    for side in (0.9, 0.3, 0.1):
        obj, gt, m = _small_object(side)
        m = m.copy(); m[gt] += 5.0
        assert aupro([gt], [m]) > 0.99, side


def test_aupro_random_map_depends_on_object_size_not_capped():
    # Background is exactly 0, so the last real threshold predicts the whole object. On a big
    # object that already costs > 30% FPR and a random map scores ~0.25; on a tiny object it
    # costs ~1% FPR and the curve continues (by tie handling) to (FPR 1, PRO 1), so even a
    # random map scores high. The old implementation stopped at the object fraction and gave
    # a tiny object ~0.0, which is what this test guards against.
    _, gt_big, m_big = _small_object(0.9, seed=1)
    _, gt_small, m_small = _small_object(0.1, seed=1)
    assert 0.1 < aupro([gt_big], [m_big]) < 0.5
    assert aupro([gt_small], [m_small]) > 0.9


def test_aupro_worst_map_is_zero():
    _, gt, m = _small_object(0.9)
    m = m.copy(); m[gt] = 0.001
    assert aupro([gt], [m]) < 0.01


def test_pixel_auroc_perfect():
    obj, gt, m = _small_object(0.3)
    m = m.copy(); m[gt] += 5.0
    assert pixel_auroc([gt], [m]) > 0.999
