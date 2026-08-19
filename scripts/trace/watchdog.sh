#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUNPOD_API_KEY:-}" || -z "${RUNPOD_POD_ID:-}" ]]; then
  exit 0
fi

watchdog_start_marker="$(mktemp /tmp/trace-watchdog-start.XXXXXX)"
trap 'rm -f "${watchdog_start_marker}"' EXIT
while true; do
  while IFS= read -r marker; do
    if [[ ! "${marker}" -nt "${watchdog_start_marker}" ]]; then
      continue
    fi
    if python -c 'import json,sys; value=json.load(open(sys.argv[1])); raise SystemExit(value.get("format") != "trace-safe-to-terminate-v1")' "${marker}"; then
      python -c 'import os,urllib.request; request=urllib.request.Request("https://rest.runpod.io/v1/pods/" + os.environ["RUNPOD_POD_ID"], method="DELETE", headers={"Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"]}); urllib.request.urlopen(request, timeout=30).close()' || true
      exit 0
    fi
  done < <(find /workspace/vamp-trace/runs -path '*/state/SAFE_TO_TERMINATE.json' -type f 2>/dev/null)
  sleep 30
done
