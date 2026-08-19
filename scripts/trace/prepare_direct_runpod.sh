#!/usr/bin/env bash
set -euo pipefail

trace_root="${TRACE_ROOT:-/workspace/vamp-trace}"
source_root="${TRACE_SOURCE_ROOT:-${trace_root}/source/apm}"
venv_root="${TRACE_VENV_ROOT:-${trace_root}/venv}"

if [[ ! -f "${source_root}/docker/trace/requirements.lock" ]]; then
  echo "TRACE source tree is missing below ${source_root}" >&2
  exit 1
fi
if [[ "$(nvidia-smi --list-gpus | wc -l)" -ne 2 ]]; then
  echo "TRACE direct deployment requires exactly two visible GPUs" >&2
  exit 1
fi

mkdir -p "${trace_root}/cache/huggingface" "${trace_root}/logs"
if [[ ! -x "${venv_root}/bin/python" ]]; then
  python -m venv "${venv_root}"
fi
"${venv_root}/bin/python" -m pip install --upgrade pip
"${venv_root}/bin/python" -m pip install \
  --requirement "${source_root}/docker/trace/requirements.lock"
"${venv_root}/bin/python" -m pip install --no-deps --editable "${source_root}"
"${venv_root}/bin/python" -m apm.continual.trace.cli self-test

echo "TRACE direct RunPod environment is ready at ${venv_root}"
