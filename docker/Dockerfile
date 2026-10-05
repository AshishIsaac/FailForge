# FailForge (Linux). Build from the repository root:
#   docker build -f docker/Dockerfile -t failforge .                      # CUDA build of PyTorch (default)
#   docker build -f docker/Dockerfile --build-arg TORCH=cpu -t failforge .
# Run (the workspace volume keeps data, models and runs between containers):
#   docker run --gpus all -p 8765:8765 -v failforge-ws:/workspace failforge
# The first start downloads the data and models into the volume (~5 GB), then serves the dashboard.
# The PyTorch CUDA wheels bundle CUDA + cuDNN: the host only needs the NVIDIA driver and the
# NVIDIA Container Toolkit.
FROM python:3.12-slim

ARG TORCH=gpu
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 FAILFORGE_HOME=/workspace
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements ./requirements
RUN pip install -r requirements/torch-${TORCH}.txt
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY reports ./reports
COPY docker/entrypoint.sh /entrypoint.sh
RUN pip install -e ".[$( [ "$TORCH" = gpu ] && echo gpu || echo cpu )]" && chmod +x /entrypoint.sh \
    && useradd -m -u 1000 ff && mkdir -p /workspace && chown ff /workspace
USER ff
VOLUME /workspace
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/health')"
ENTRYPOINT ["/entrypoint.sh"]
CMD ["serve", "--host", "0.0.0.0", "--no-browser"]
