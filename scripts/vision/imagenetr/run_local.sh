#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
vision_python="${project_root}/.venv-vision/bin/python"
config="${project_root}/configs/vision/imagenetr/primary.yaml"

if [[ ! -x "${vision_python}" ]]; then
  echo "Missing isolated environment. Run scripts/vision/imagenetr/bootstrap.sh first." >&2
  exit 2
fi

exec "${vision_python}" -m apm.continual.vision.imagenetr.cli run "${config}"
