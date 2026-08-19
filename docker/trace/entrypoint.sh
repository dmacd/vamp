#!/usr/bin/env bash
set -euo pipefail

/start.sh &
/opt/apm/scripts/trace/watchdog.sh &

if (( $# > 0 )); then
  exec "$@"
fi
exec /opt/apm/scripts/trace/run_primary.sh
