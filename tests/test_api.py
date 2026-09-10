import io

import numpy as np
import tifffile
from fastapi.testclient import TestClient

from pcinspect.inspector import Inspector
from tests.synthetic import make_scan


def _tiff_bytes(xyz: np.ndarray) -> bytes:
    buf = io.BytesIO()
    tifffile.imwrite(buf, xyz)
    return buf.getvalue()


def test_api_round_trip(tmp_path, monkeypatch):
    insp = Inspector.fit([make_scan(seed=s)[0] for s in range(3)])
    insp.calibrate([make_scan(seed=s)[0] for s in range(10, 13)])
    insp.save(tmp_path / "dome")

    from pcinspect.api import app as api
    monkeypatch.setattr(api, "MODELS_DIR", tmp_path)
    api._inspectors.clear()
    client = TestClient(api.app)

    assert client.get("/health").json()["categories"] == ["dome"]
    assert "dome" in client.get("/categories").json()

    bad_xyz, _ = make_scan(dent=True, seed=7)
    r = client.post("/inspect", params={"category": "dome"},
                    files={"file": ("scan.tiff", _tiff_bytes(bad_xyz), "image/tiff")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["is_defective"] is True
    assert body["heatmap_png_base64"]

    r = client.post("/inspect/heatmap.png", params={"category": "dome"},
                    files={"file": ("scan.tiff", _tiff_bytes(bad_xyz), "image/tiff")})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"

    r = client.post("/inspect", params={"category": "nope"},
                    files={"file": ("scan.tiff", _tiff_bytes(bad_xyz), "image/tiff")})
    assert r.status_code == 404

    r = client.post("/inspect", params={"category": "dome"},
                    files={"file": ("scan.txt", b"hello", "text/plain")})
    assert r.status_code == 415
