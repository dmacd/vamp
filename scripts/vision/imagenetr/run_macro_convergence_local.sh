#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
exec .venv-vision/bin/python -m apm.continual.vision.imagenetr.macro_convergence_workflow
