#!/usr/bin/env bash
set -euo pipefail

export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec nice -n 10 .venv-vision/bin/python -m apm.continual.vision.imagenetr.srt_followup "$@"
