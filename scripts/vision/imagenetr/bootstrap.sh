#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
vision_environment="${project_root}/.venv-vision"

echo "Phase 1/3: creating the isolated ImageNet-R vision environment."
python3.10 -m venv "${vision_environment}"

echo "Phase 2/3: installing the pinned CUDA 12.1 PyTorch pair."
"${vision_environment}/bin/python" -m pip install --upgrade "pip==25.2"
"${vision_environment}/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.5.1" "torchvision==0.20.1"

echo "Phase 3/3: installing the pinned vision and test dependencies."
"${vision_environment}/bin/python" -m pip install -e "${project_root}[vision,dev]"
"${vision_environment}/bin/python" -m pip check

echo "Vision environment ready: ${vision_environment}"
