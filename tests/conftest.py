import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app as app_module
from cnnClassifier.utils.model_loader import LoadedModel


class FakeModel:
    """Stands in for the real Keras model, so tests need no model file or network."""

    def __init__(self, probabilities):
        self.probabilities = np.array([probabilities])

    def predict(self, _batch):
        return self.probabilities


@pytest.fixture(autouse=True)
def fresh_rate_limiter():
    """Rate-limit counters live in memory, so start every test from zero."""
    app_module.limiter.reset()


@pytest.fixture
def fake_model():
    """Factory: fake_model([0.1, 0.9]) -> a model that always answers with these odds."""
    return FakeModel


@pytest.fixture
def make_client(monkeypatch):
    """Factory for a started API (lifespan runs) whose model loader is faked.

    Usage: client = make_client([0.1, 0.9])   # model says: 10% class 0, 90% class 1
    """
    clients = []

    def _make(probabilities=(0.1, 0.9)):
        monkeypatch.setattr(
            app_module,
            "load_serving_model",
            lambda: LoadedModel(FakeModel(probabilities), "fake-model", "7"),
        )
        client = TestClient(app_module.app)
        client.__enter__()  # starts the app, which runs the lifespan hook
        clients.append(client)
        return client

    yield _make

    for client in clients:
        client.__exit__(None, None, None)


def image_bytes(image_format="PNG", size=(64, 64), mode="RGB"):
    buffer = io.BytesIO()
    Image.new(mode, size, color=1 if mode == "1" else (120, 120, 120)).save(
        buffer, format=image_format
    )
    return buffer.getvalue()


@pytest.fixture
def png_bytes():
    """A small valid PNG image, built in memory."""
    return image_bytes("PNG")


@pytest.fixture
def png_path(tmp_path, png_bytes):
    path = tmp_path / "scan.png"
    path.write_bytes(png_bytes)
    return str(path)
