import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from cnnClassifier import logger
from cnnClassifier.pipeline.prediction import PredictionPipeline

# run everything relative to the project root, so paths like "model/model.h5" work
ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)

TRAINED_MODEL_PATH = ROOT_DIR / "artifacts" / "training" / "model.h5"
SERVING_MODEL_PATH = ROOT_DIR / "model" / "model.h5"
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}

app = FastAPI(
    title="Chest Cancer Classification API",
    description="Classifies chest CT scans as Normal or Adenocarcinoma Cancer.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def promote_model() -> None:
    """Copy the latest trained model from artifacts/ to model/, which the API serves from."""
    if not TRAINED_MODEL_PATH.exists():
        raise FileNotFoundError(f"No trained model found at {TRAINED_MODEL_PATH}")
    SERVING_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TRAINED_MODEL_PATH, SERVING_MODEL_PATH)
    logger.info(f"Model promoted: {TRAINED_MODEL_PATH} -> {SERVING_MODEL_PATH}")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(ROOT_DIR / "templates" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_available": SERVING_MODEL_PATH.exists()}


@app.post("/train")
def train():
    """Re-run the DVC pipeline (only changed stages run), then promote the new model."""
    try:
        # call dvc through this Python, so it works even if the venv isn't activated
        subprocess.run([sys.executable, "-m", "dvc", "repro"], check=True)
        promote_model()
    except subprocess.CalledProcessError as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail="Training pipeline failed") from e
    except FileNotFoundError as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"message": "Training done successfully, model promoted to model/"}


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File()]):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Upload a JPG or PNG image")
    if not SERVING_MODEL_PATH.exists():
        raise HTTPException(
            status_code=503, detail="No model in model/. Run /train first."
        )

    # save the upload to a temp file, since the pipeline reads images from disk
    suffix = Path(file.filename or "").suffix or ".jpg"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(await file.read())
        result = PredictionPipeline(tmp_path).predict()
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail="Prediction failed") from e
    finally:
        os.remove(tmp_path)

    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
