# Hybrid CPU/GPU image. Pass `--build-arg GPU=1` (or use docker-compose.gpu.yml)
# to swap the torch/tf wheels for CUDA-enabled ones after the base sync.
#
# CPU path (default): every dep comes from uv.lock, including the CPU torch wheel.
# GPU path: post-sync pip-installs CUDA wheels over the CPU ones via PyTorch's
# extra-index URL. We do this AFTER the lock-sync so the lock stays GPU-agnostic.
FROM python:3.11-slim AS builder

ARG GPU=0

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

COPY --from=ghcr.io/astral-sh/uv:0.6.8 /uv /uvx /bin/

WORKDIR /prod

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY deepCab deepCab
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Swap to CUDA wheels when requested. Keeps the default image lean; only GPU
# builds pay the ~2GB torch-cu121 download.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$GPU" = "1" ]; then \
        uv pip install --upgrade \
            --index-url https://download.pytorch.org/whl/cu121 \
            torch ; \
    fi

FROM python:3.11-slim AS runtime

WORKDIR /prod

COPY --from=builder /prod /prod
ENV PATH="/prod/.venv/bin:$PATH"

CMD uvicorn deepCab.api.app:app --host 0.0.0.0 --port ${PORT:-8000}
