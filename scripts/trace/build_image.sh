#!/usr/bin/env bash
set -euo pipefail

: "${TRACE_IMAGE:?Set TRACE_IMAGE to an immutable registry tag}"

docker build \
  --platform=linux/amd64 \
  --file docker/trace/Dockerfile \
  --tag "${TRACE_IMAGE}" \
  .
docker push "${TRACE_IMAGE}"
