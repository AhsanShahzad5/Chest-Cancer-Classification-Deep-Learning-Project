import os
import re
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from cnnClassifier import logger, request_id_var
from cnnClassifier.entity.schemas import HealthResponse, PredictionResponse
from cnnClassifier.pipeline.prediction import PredictionPipeline
from cnnClassifier.utils.image_validation import (
    MAX_UPLOAD_BYTES,
    ImageValidationError,
    validate_image,
)
from cnnClassifier.utils.model_loader import load_serving_model

# run everything relative to the project root, so relative paths work anywhere
ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)

# how many predictions one client (IP address) may request, e.g. "10/minute"
RATE_LIMIT = os.getenv("RATE_LIMIT", "10/minute")

REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")
QUIET_PATHS = {"/health", "/ready"}  # polled constantly, so logged at DEBUG only

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts (before it accepts requests) and once at shutdown."""
    # If this raises (MLflow unreachable, alias missing...), the server refuses to
    # start. That is deliberate: a deploy that can't serve predictions should fail
    # loudly instead of looking healthy.
    loaded = load_serving_model()
    app.state.model = loaded.model
    app.state.model_source = loaded.source
    app.state.model_version = loaded.version
    logger.info(f"Serving model from: {loaded.source} (version {loaded.version})")
    yield
    app.state.model = None


app = FastAPI(
    title="Chest Cancer Classification API",
    description="Classifies chest CT scans as Normal or Adenocarcinoma Cancer.",
    version="0.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter


async def too_many_requests(_request: Request, _exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please wait a minute and try again."},
        headers={"Retry-After": "60"},
    )


app.add_exception_handler(RateLimitExceeded, too_many_requests)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Give every request an ID and write one log line when it finishes."""
    incoming = request.headers.get("X-Request-ID", "")
    request_id = incoming if REQUEST_ID_PATTERN.fullmatch(incoming) else uuid.uuid4().hex[:16]
    token = request_id_var.set(request_id)
    started = time.perf_counter()
    status_code = 500  # what gets logged if the request crashes
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        log = logger.debug if request.url.path in QUIET_PATHS else logger.info
        log(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": status_code,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        request_id_var.reset(token)


def model_status(request: Request, status: str) -> HealthResponse:
    state = request.app.state
    return HealthResponse(
        status=status,
        model_loaded=getattr(state, "model", None) is not None,
        model_source=getattr(state, "model_source", None),
        model_version=getattr(state, "model_version", None),
    )


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(ROOT_DIR / "templates" / "index.html")


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    """Liveness: the process is up. Says nothing about whether it can predict."""
    return model_status(request, "ok")


@app.get("/ready", response_model=HealthResponse)
def ready(request: Request):
    """Readiness: 200 only when a model is loaded and predictions can be served."""
    report = model_status(request, "ready")
    if not report.model_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return report


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit(RATE_LIMIT)
async def predict(request: Request, file: Annotated[UploadFile, File()]):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    # read one byte past the limit, so oversized files are caught without loading them whole
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    try:
        validate_image(data)
    except ImageValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message) from e

    # save the upload to a temp file, since the pipeline reads images from disk
    suffix = Path(file.filename or "").suffix or ".jpg"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        result = PredictionPipeline(model).predict(tmp_path)
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail="Prediction failed") from e
    finally:
        os.remove(tmp_path)

    return PredictionResponse(**result, model_version=request.app.state.model_version)


if __name__ == "__main__":
    # log_config=None lets uvicorn's own messages use our JSON logger,
    # and access_log is off because log_requests already logs every request
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None, access_log=False)
