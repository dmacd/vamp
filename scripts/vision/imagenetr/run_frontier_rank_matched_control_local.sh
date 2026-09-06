#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec nice -n 10 .venv-vision/bin/python -m apm.continual.vision.imagenetr.frontier_rank_matched_workflow

