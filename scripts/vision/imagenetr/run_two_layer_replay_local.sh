#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
exec .venv-vision/bin/python -m apm.continual.vision.imagenetr.two_layer_replay_workflow "$@"
