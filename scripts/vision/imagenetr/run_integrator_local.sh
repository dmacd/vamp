#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$project_root"

exec .venv-vision/bin/python -m apm.continual.vision.imagenetr.integrator_cli \
  "${1:-run}" \
  "${2:-configs/vision/imagenetr/logt_prediction_integrator_full_union_ungated_v3.yaml}"
