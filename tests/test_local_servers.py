"""Local GPU mirrors: contract checks that need no model (nothing is loaded on the GPU)."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from vastkit.config import Settings
from vastkit.gpu import CosmosEmbed, CosmosReason
from vastkit.local import cosmos_edge_server, embed_server, yolo_server


@pytest.mark.parametrize("server, make", [
    (cosmos_edge_server, lambda: CosmosReason("http://testserver", model="m", cfg=Settings(), cache=False)),
    (embed_server, lambda: CosmosEmbed("http://testserver", cfg=Settings())),
])
def test_ready_follows_health_ready(server, make, monkeypatch):
    c = make()
    c.client = TestClient(server.app)
    assert c.client.get("/v1/models").status_code in (200, 401)
    assert c.ready() is False  # 503 while loading (or after a failed load), whatever /v1/models says
    monkeypatch.setitem(server._state, "model", object())
    assert c.ready() is True


class _L(list):
    def tolist(self) -> list:
        return list(self)


def test_yolo_source_matches_hosted(monkeypatch, tmp_path):
    boxes = SimpleNamespace(cls=_L([0.0]), conf=_L([0.9]), xyxy=_L([[1.0, 2.0, 3.0, 4.0]]))
    frame = SimpleNamespace(boxes=boxes, names={0: "person"}, orig_shape=(720, 1280))
    monkeypatch.setattr(yolo_server, "_model", SimpleNamespace(predict=lambda *a, **k: iter([frame, frame])))
    monkeypatch.setattr(yolo_server, "WEIGHTS", "yolo26s.engine")  # the source must not follow the weights name
    monkeypatch.delenv("PERCEPTION_SOURCE", raising=False)
    out = yolo_server.run_infer(tmp_path / "c.mp4", 0.4, 1, False)
    assert out["perception_json"]["source"] == "yolo11_coco" and out["object_counts"] == {"person": 2}
    monkeypatch.setenv("PERCEPTION_SOURCE", "custom_source")
    assert yolo_server.run_infer(tmp_path / "c.mp4", 0.4, 1, False)["perception_json"]["source"] == "custom_source"
