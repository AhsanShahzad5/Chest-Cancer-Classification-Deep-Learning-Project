import os
import re
from typing import NamedTuple

from dotenv import load_dotenv

from cnnClassifier import logger
from cnnClassifier.constants import DEFAULT_MODEL_URI, LOCAL_MODEL_PATH


class LoadedModel(NamedTuple):
    model: object  # the Keras model
    source: str  # where it came from, e.g. "models:/VGG16Model@champion"
    version: str | None  # registry version number, e.g. "4" (None if unknown)


def load_serving_model() -> LoadedModel:
    """Load the model the API should serve.

    Chosen by the MODEL_SOURCE environment variable:
      - "mlflow" (default): download the registry model MODEL_URI from MLflow
      - "local": read a .h5 file from disk, for offline development
    """
    load_dotenv()  # lets local runs pick up .env; a real environment variable wins
    source = os.getenv("MODEL_SOURCE", "mlflow").strip().lower()

    if source == "mlflow":
        return _load_from_mlflow()
    if source == "local":
        return _load_from_disk()
    raise ValueError(f"MODEL_SOURCE must be 'mlflow' or 'local', got '{source}'")


def _load_from_mlflow() -> LoadedModel:
    # imported here so local mode and tests don't pay for importing mlflow
    import mlflow
    from mlflow.keras import load_model

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is not set")
    # MLFLOW_TRACKING_USERNAME / _PASSWORD are read by mlflow from the environment
    mlflow.set_tracking_uri(tracking_uri)

    model_uri = os.getenv("MODEL_URI", DEFAULT_MODEL_URI)
    logger.info(f"Loading model {model_uri} from MLflow")
    model = load_model(model_uri)
    version = _resolve_version(model_uri)
    logger.info(f"Model loaded from MLflow: {model_uri} (version {version})")
    return LoadedModel(model, model_uri, version)


def _resolve_version(model_uri: str) -> str | None:
    """Turn 'models:/Name@alias' or 'models:/Name/3' into a version number."""
    by_version = re.fullmatch(r"models:/([^/@]+)/(\d+)", model_uri)
    if by_version:
        return by_version.group(2)

    by_alias = re.fullmatch(r"models:/([^/@]+)@(.+)", model_uri)
    if by_alias:
        try:
            from mlflow import MlflowClient

            found = MlflowClient().get_model_version_by_alias(by_alias.group(1), by_alias.group(2))
            return str(found.version)
        except Exception as e:  # the model is already loaded; a missing label is not fatal
            logger.warning(f"Could not resolve the version behind {model_uri}: {e}")
    return None


def _load_from_disk() -> LoadedModel:
    from keras.models import load_model

    path = os.getenv("LOCAL_MODEL_PATH", str(LOCAL_MODEL_PATH))
    if not os.path.exists(path):
        raise FileNotFoundError(f"MODEL_SOURCE=local but no model file at {path}")
    logger.info(f"Loading model from {path}")
    model = load_model(path)
    logger.info(f"Model loaded from disk: {path}")
    return LoadedModel(model, f"local:{path}", "local")
