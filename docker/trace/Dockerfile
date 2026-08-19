FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/vamp-trace/cache/huggingface \
    HF_DATASETS_CACHE=/workspace/vamp-trace/cache/huggingface/datasets \
    TRANSFORMERS_CACHE=/workspace/vamp-trace/cache/huggingface/models

WORKDIR /opt/apm

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY docker/trace/requirements.lock /tmp/trace-requirements.lock
RUN python -m pip uninstall -y torchaudio torchvision
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-cache-dir -r /tmp/trace-requirements.lock

COPY pyproject.toml README.md /opt/apm/
COPY src /opt/apm/src
COPY configs/trace /opt/apm/configs/trace
COPY docker/trace /opt/apm/docker/trace
COPY scripts/trace /opt/apm/scripts/trace
RUN python -m pip install --no-deps /opt/apm \
    && chmod +x /opt/apm/docker/trace/entrypoint.sh /opt/apm/scripts/trace/*.sh \
    && python -m apm.continual.trace.cli self-test

# Start the official SSH/web-terminal services alongside the resumable experiment.
ENTRYPOINT ["/opt/apm/docker/trace/entrypoint.sh"]
