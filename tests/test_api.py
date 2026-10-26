import pytest
from fastapi.testclient import TestClient

import app as app_module
from cnnClassifier.utils.image_validation import MAX_UPLOAD_BYTES
from cnnClassifier.utils.model_loader import LoadedModel
from tests.conftest import image_bytes


def post_image(client, data, content_type="image/png", name="scan.png"):
    return client.post("/predict", files={"file": (name, data, content_type)})


# ---------- pages and status endpoints ----------


def test_home_page_loads(make_client):
    response = make_client().get("/")
    assert response.status_code == 200
    assert "Chest Cancer Classifier" in response.text


def test_health_reports_loaded_model(make_client):
    body = make_client().get("/health").json()
    assert body == {
        "status": "ok",
        "model_loaded": True,
        "model_source": "fake-model",
        "model_version": "7",
    }


def test_ready_when_model_loaded(make_client):
    response = make_client().get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_not_ready_but_still_alive_without_model(make_client):
    client = make_client()
    client.app.state.model = None
    assert client.get("/ready").status_code == 503
    health = client.get("/health")
    assert health.status_code == 200  # alive...
    assert health.json()["model_loaded"] is False  # ...but cannot predict


def test_train_endpoint_is_gone(make_client):
    assert make_client().post("/train").status_code == 404


def test_openapi_documents_response_shapes(make_client):
    schemas = make_client().get("/openapi.json").json()["components"]["schemas"]
    assert {"PredictionResponse", "HealthResponse"} <= set(schemas)


# ---------- predictions ----------


def test_predict_returns_normal(make_client, png_bytes):
    response = post_image(make_client([0.1, 0.9]), png_bytes)
    assert response.status_code == 200
    assert response.json() == {
        "prediction": "Normal",
        "confidence": 0.9,
        "model_version": "7",
    }


def test_predict_returns_cancer(make_client, png_bytes):
    response = post_image(make_client([0.8, 0.2]), png_bytes)
    assert response.status_code == 200
    assert response.json()["prediction"] == "Adenocarcinoma Cancer"
    assert response.json()["confidence"] == 0.8


def test_predict_accepts_jpeg(make_client):
    response = post_image(make_client(), image_bytes("JPEG"), "image/jpeg", "scan.jpg")
    assert response.status_code == 200


def test_predict_returns_503_when_model_missing(make_client, png_bytes):
    client = make_client()
    client.app.state.model = None
    assert post_image(client, png_bytes).status_code == 503


# ---------- upload validation ----------


def test_rejects_text_file(make_client):
    response = post_image(make_client(), b"hello", "text/plain", "notes.txt")
    assert response.status_code == 400


def test_rejects_text_pretending_to_be_png(make_client):
    # the header says image/png, but the bytes are not an image
    response = post_image(make_client(), b"this is not an image", "image/png")
    assert response.status_code == 400
    assert "not a valid image" in response.json()["detail"]


def test_rejects_unsupported_image_format(make_client):
    response = post_image(make_client(), image_bytes("GIF"), "image/gif", "scan.gif")
    assert response.status_code == 400
    assert "JPEG and PNG" in response.json()["detail"]


def test_rejects_empty_file(make_client):
    assert post_image(make_client(), b"").status_code == 400


def test_rejects_oversized_file(make_client):
    response = post_image(make_client(), b"x" * (MAX_UPLOAD_BYTES + 10))
    assert response.status_code == 413


def test_rejects_huge_dimensions(make_client):
    # a 1-bit 5000x5000 PNG is only a few KB on disk but 25 million pixels
    huge = image_bytes("PNG", size=(5000, 5000), mode="1")
    assert len(huge) < MAX_UPLOAD_BYTES
    assert post_image(make_client(), huge).status_code == 422


# ---------- rate limiting ----------


def test_rate_limit_returns_429_after_limit(make_client, png_bytes):
    client = make_client()
    allowed = int(app_module.RATE_LIMIT.split("/")[0])
    for _ in range(allowed):
        assert post_image(client, png_bytes).status_code == 200

    blocked = post_image(client, png_bytes)
    assert blocked.status_code == 429
    assert blocked.headers["Retry-After"] == "60"
    assert "Too many requests" in blocked.json()["detail"]


def test_rate_limit_does_not_apply_to_health(make_client):
    client = make_client()
    for _ in range(30):
        assert client.get("/health").status_code == 200


# ---------- startup behaviour ----------


def test_model_loaded_once_not_per_request(monkeypatch, png_bytes, fake_model):
    calls = []

    def counting_loader():
        calls.append(1)
        return LoadedModel(fake_model([0.1, 0.9]), "fake-model", "7")

    monkeypatch.setattr(app_module, "load_serving_model", counting_loader)
    with TestClient(app_module.app) as client:
        for _ in range(3):
            assert post_image(client, png_bytes).status_code == 200
    assert len(calls) == 1


def test_server_refuses_to_start_if_model_cannot_load(monkeypatch):
    def broken_loader():
        raise RuntimeError("MLflow unreachable")

    monkeypatch.setattr(app_module, "load_serving_model", broken_loader)
    with pytest.raises(RuntimeError, match="MLflow unreachable"), TestClient(app_module.app):
        pass
