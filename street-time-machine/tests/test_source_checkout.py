"""Source checkout smoke checks; no videos, network, model loads or inference."""
import json

from fastapi.testclient import TestClient

from stm import data, server


def test_synthetic_demo_can_be_served_without_private_data(monkeypatch):
    sample = data.ROOT / "examples" / "synthetic"
    monkeypatch.setattr(server, "OUT", sample)
    client = TestClient(server.app)
    cameras = client.get("/api/cams").json()
    assert [c["camera"] for c in cameras] == ["synthetic_demo"]
    assert cameras[0]["simulated"] and cameras[0]["ask"] == "precomputed"
    assert client.get("/").status_code == 200
    assert client.get("/out/synthetic_demo/meta.json").status_code == 200
    scene = client.get("/out/synthetic_demo/scene.svg")
    assert scene.status_code == 200 and "SYNTHETIC EXAMPLE" in scene.text
    question = json.loads((sample / "synthetic_demo" / "qa.json").read_text())["items"][0]["question"]
    result = client.get("/api/ask", params={"cam": "synthetic_demo", "q": question})
    assert result.status_code == 200
    assert result.headers["x-stm-source"] == "precomputed"
    assert result.json()["latency_ms"] is None
    assert "synthetic" in result.json()["honesty"].lower()


def test_source_checkout_paths_are_portable():
    assert data.ROOT in data.TRACK_CACHE.parents
    assert data.ROOT in data.VIDEO_ROOT.parents
    config = json.loads((data.ROOT / "cams.example.json").read_text())
    assert not __import__("pathlib").Path(config["my_camera"]["video"]).is_absolute()
