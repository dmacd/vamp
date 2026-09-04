#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$project_root"

.venv-vision/bin/python -m apm.continual.vision.imagenetr.parent_recipe_factorial
.venv-vision/bin/python -m apm.continual.vision.imagenetr.parent_recipe_reporting
