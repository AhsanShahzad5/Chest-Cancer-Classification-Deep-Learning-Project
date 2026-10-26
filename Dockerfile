FROM python:3.11-slim

# uv for installing dependencies from uv.lock
COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /usr/local/bin/uv

WORKDIR /app

# install dependencies first, so this layer is cached when only code changes
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --no-cache

# copy the code (and model/model.h5), then install the project itself
COPY . .
RUN uv sync --frozen --no-dev --no-cache

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

EXPOSE 8000

CMD ["python", "app.py"]
