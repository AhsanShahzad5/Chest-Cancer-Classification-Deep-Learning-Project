import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app as app_module
from cnnClassifier.pipeline import prediction


class FakeModel:
    """Stands in for the real Keras model, so tests need no model file."""

    def __init__(self, probabilities):
        self.probabilities = np.array([probabilities])

    def predict(self, _batch):
        return self.probabilities


@pytest.fixture
def client():
    return TestClient(app_module.app)


@pytest.fixture
def png_bytes():
    """A small valid PNG image, built in memory."""
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color=(120, 120, 120)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def png_path(tmp_path, png_bytes):
    path = tmp_path / "scan.png"
    path.write_bytes(png_bytes)
    return str(path)


@pytest.fixture
def use_fake_model(monkeypatch, tmp_path):
    """Returns a function that makes the app use a fake model with given probabilities."""

    def _use(probabilities):
        fake_model_file = tmp_path / "model.h5"
        fake_model_file.write_bytes(b"not a real model")
        monkeypatch.setattr(app_module, "SERVING_MODEL_PATH", fake_model_file)
        monkeypatch.setattr(
            prediction, "load_model", lambda _path: FakeModel(probabilities)
        )

    return _use
