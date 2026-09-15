#!/bin/sh
# Regenerate checkpoints/dashboard.html every 5s. The HTML itself
# auto-refreshes (meta tag), so just open it once in a browser and leave the
# tab open — no server needed, doesn't touch the running `train` process.
cd "$(dirname "$0")/.."
while true; do
  uv run scripts/dashboard.py >/dev/null 2>&1
  sleep 5
done
