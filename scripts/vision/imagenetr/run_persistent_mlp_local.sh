#!/usr/bin/env bash
set -euo pipefail

exec nice -n 10 .venv-vision/bin/python -m apm.continual.vision.imagenetr.persistent_mlp_workflow
