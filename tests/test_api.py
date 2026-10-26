def test_home_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Chest Cancer Classifier" in response.text


def test_health_is_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_rejects_non_image(client):
    response = client.post(
        "/predict", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 400


def test_predict_without_model_returns_503(client, png_bytes, monkeypatch, tmp_path):
    import app as app_module

    monkeypatch.setattr(app_module, "SERVING_MODEL_PATH", tmp_path / "missing.h5")
    response = client.post("/predict", files={"file": ("scan.png", png_bytes, "image/png")})
    assert response.status_code == 503


def test_predict_returns_normal(client, png_bytes, use_fake_model):
    use_fake_model([0.1, 0.9])
    response = client.post("/predict", files={"file": ("scan.png", png_bytes, "image/png")})
    assert response.status_code == 200
    assert response.json() == [{"image": "Normal"}]


def test_predict_returns_cancer(client, png_bytes, use_fake_model):
    use_fake_model([0.9, 0.1])
    response = client.post("/predict", files={"file": ("scan.png", png_bytes, "image/png")})
    assert response.status_code == 200
    assert response.json() == [{"image": "Adenocarcinoma Cancer"}]
