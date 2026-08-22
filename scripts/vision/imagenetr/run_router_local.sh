#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$project_root"

exec .venv-vision/bin/python -m apm.continual.vision.imagenetr.router_cli \
  "${1:-run}" \
  "${2:-configs/vision/imagenetr/recursive_router_oracle_recovery_v1.yaml}"
