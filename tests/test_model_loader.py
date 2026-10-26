import pytest

from cnnClassifier.utils import model_loader


@pytest.fixture(autouse=True)
def ignore_dotenv(monkeypatch):
    """Don't let a developer's real .env leak into these tests."""
    monkeypatch.setattr(model_loader, "load_dotenv", lambda: None)


def test_unknown_source_is_rejected(monkeypatch):
    monkeypatch.setenv("MODEL_SOURCE", "somewhere-else")
    with pytest.raises(ValueError, match="MODEL_SOURCE"):
        model_loader.load_serving_model()


def test_local_source_without_file_fails_clearly(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_SOURCE", "local")
    monkeypatch.setenv("LOCAL_MODEL_PATH", str(tmp_path / "missing.h5"))
    with pytest.raises(FileNotFoundError, match="no model file"):
        model_loader.load_serving_model()


def test_mlflow_source_requires_tracking_uri(monkeypatch):
    monkeypatch.setenv("MODEL_SOURCE", "mlflow")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI"):
        model_loader.load_serving_model()


def test_mlflow_is_the_default_source(monkeypatch):
    monkeypatch.delenv("MODEL_SOURCE", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI"):
        model_loader.load_serving_model()


# ---------- which registry version is being served ----------


def test_version_is_read_from_a_pinned_uri():
    assert model_loader._resolve_version("models:/VGG16Model/3") == "3"


def test_version_is_looked_up_from_an_alias(monkeypatch):
    class FakeClient:
        def get_model_version_by_alias(self, name, alias):
            assert (name, alias) == ("VGG16Model", "champion")
            return type("Version", (), {"version": 4})()

    monkeypatch.setattr("mlflow.MlflowClient", FakeClient)
    assert model_loader._resolve_version("models:/VGG16Model@champion") == "4"


def test_version_lookup_failure_is_not_fatal(monkeypatch):
    class BrokenClient:
        def get_model_version_by_alias(self, name, alias):
            raise RuntimeError("registry offline")

    monkeypatch.setattr("mlflow.MlflowClient", BrokenClient)
    assert model_loader._resolve_version("models:/VGG16Model@champion") is None


def test_version_is_unknown_for_other_uris():
    assert model_loader._resolve_version("runs:/abc123/model") is None
