"""Helpers for retraining. They are NOT exposed through the API.

Run them from a terminal or a CI job from the project root, e.g.:
    uv run python -c "from cnnClassifier.utils.training import run_training_pipeline; run_training_pipeline()"
"""

import shutil
import subprocess
import sys
from pathlib import Path

from cnnClassifier import logger
from cnnClassifier.constants import LOCAL_MODEL_PATH, TRAINED_MODEL_PATH


def run_training_pipeline() -> None:
    """Run the DVC pipeline. DVC only re-runs the stages whose inputs changed."""
    # call dvc through the current Python, so it works without an activated venv
    subprocess.run([sys.executable, "-m", "dvc", "repro"], check=True)


def promote_trained_model() -> Path:
    """Copy the freshly trained model to the local serving folder (MODEL_SOURCE=local)."""
    if not TRAINED_MODEL_PATH.exists():
        raise FileNotFoundError(f"No trained model found at {TRAINED_MODEL_PATH}")
    LOCAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TRAINED_MODEL_PATH, LOCAL_MODEL_PATH)
    logger.info(f"Model promoted: {TRAINED_MODEL_PATH} -> {LOCAL_MODEL_PATH}")
    return LOCAL_MODEL_PATH
