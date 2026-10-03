#!/bin/bash
# run_capture.sh -- drive one StreetLab capture to completion, end to end,
# as a single blocking foreground command.
#
# Why this exists (Phase 3b Task 2 solved this once, threw the driver away
# as "scratch", and Task 3 had to rediscover it twice -- once from a stray
# scratchpad copy, once for real after a stranded background wait cost a
# dispatch). It is committed here so the eleven-capture spend, and any
# future capture, can point at one reproducible script instead of prose in
# a report.
#
# THE FAILURE THIS WORKS AROUND: `server/cli.py`'s `_start_stdin_watchdog`
# blocks on `sys.stdin.read()` and calls `os._exit(0)` the moment that
# returns. A backend launched in the background (or with stdin closed /
# redirected from /dev/null) gets EOF immediately, so it prints
# STREETLAB_READY and exits within about a second -- long before any frame
# arrives. The fix is to give it a stdin that blocks forever without ever
# closing: `< <(sleep 99999)`.
#
# THE OTHER FAILURE THIS WORKS AROUND: arming a background process and then
# idly waiting on it (a Monitor, a bare `wait`, a poll loop issued as its
# own backgrounded command) has twice lost the backend, Vite and the
# Playwright driver simultaneously mid-capture, with no error in any of
# their logs -- consistent with an idle-session reaper, not an application
# crash. The fix is to fold backend + Vite + the Playwright driver + the
# frame-count poll into ONE foreground script that blocks until it is
# itself done and then returns control -- nothing is ever left running
# unattended in the background across a turn boundary.
#
# Usage:
#   scripts/run_capture.sh <scenario> <seed> <traffic> <out_dir> [target] [maxwait_s]
#
#   scenario    e.g. grid-loop, grid-arterial, grid-signals, grid-night
#   seed        integer seed
#   traffic     --traffic value passed to `streetlab serve`
#   out_dir     capture directory (frames/ + labels.json land here; NOT
#               committed -- only the manifest built from it is)
#   target      frames to capture before stopping (default 150; the brief's
#               band is 150-250, and 150 keeps the whole run inside one
#               command's timeout budget)
#   maxwait_s   seconds to poll before giving up and stopping anyway
#               (default 480)
#
# Environment:
#   DETECTOR_MODEL   path to the .onnx detector. Defaults to the author's
#                    weights-cache path, which is the one every Phase 3b
#                    capture actually used -- but it is a default, not a
#                    constant, so this script runs on a machine that is not
#                    that laptop. `streetlab serve --detector-model` resolves
#                    the shipped model through the cache when the path is
#                    omitted entirely; this script always passes one, so point
#                    DETECTOR_MODEL at your own cache to reproduce a capture.
#
# Cleanup is scoped to THIS repo. `pkill -f` for Vite matches the frontend's
# own absolute `node_modules/.bin/vite` path rather than the bare string
# "vite", because a bare match kills any unrelated Vite dev server on the
# machine -- and there routinely is one. The `npm run dev` parent is killed
# by PID; this pkill exists only to reap the vite child it spawns.
#
# Stops the backend with `kill -INT` first, escalating to `-TERM` only if
# it has not exited 20s later: a prior phase measured that TERM leaves
# labels.json stale mid-write, while INT reaches `finalize()`, so INT is
# always tried first and given the chance to shut down cleanly -- TERM is
# a last-resort fallback for a hung process, not the normal path. Verifies
# labels.json's image count against frames on disk before exiting, and
# fails loudly (nonzero exit, clear message) if they disagree -- a silent
# mismatch would poison a manifest built from this capture.
set -uo pipefail

SCENARIO="${1:?scenario required}"
SEED="${2:?seed required}"
TRAFFIC="${3:?traffic required}"
CAPTURE_DIR="${4:?out_dir required}"
TARGET="${5:-150}"
MAXWAIT="${6:-480}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$REPO_ROOT/streetlab-backend"
FRONTEND_DIR="$REPO_ROOT/streetlab"
LOGDIR=/tmp/streetlab-capture
DETECTOR_MODEL="${DETECTOR_MODEL:-/Users/jasonpereira/Library/Caches/StreetLab/models/rtdetr_r18vd_quantized-85703b0f56dbaceb.onnx}"

mkdir -p "$LOGDIR"
rm -rf "$CAPTURE_DIR"
mkdir -p "$CAPTURE_DIR"

echo "=== cleaning up any stale processes from a prior run ==="
pkill -f "streetlab serve" 2>/dev/null
pkill -f "drive_capture" 2>/dev/null
pkill -f "$FRONTEND_DIR/node_modules/.bin/vite" 2>/dev/null
sleep 2

echo "=== starting backend: scenario=$SCENARIO seed=$SEED traffic=$TRAFFIC -> $CAPTURE_DIR ==="
cd "$BACKEND_DIR"
# stdin held open by a long-lived `sleep` -- see header. Never plain
# background / </dev/null, or the stdin watchdog kills this within ~1s.
uv run streetlab serve --port 8765 --scenario "$SCENARIO" --seed "$SEED" --traffic "$TRAFFIC" \
  --perception ml \
  --detector-model "$DETECTOR_MODEL" \
  --capture "$CAPTURE_DIR" < <(sleep 99999) > "$LOGDIR/backend.log" 2>&1 &
BACKEND_PID=$!
echo "backend pid $BACKEND_PID"

for i in $(seq 1 30); do
  grep -q STREETLAB_READY "$LOGDIR/backend.log" 2>/dev/null && break
  sleep 1
done
if ! grep -q STREETLAB_READY "$LOGDIR/backend.log" 2>/dev/null; then
  echo "FATAL: backend never printed STREETLAB_READY"
  cat "$LOGDIR/backend.log"
  kill -INT "$BACKEND_PID" 2>/dev/null
  exit 1
fi

echo "=== starting vite ==="
cd "$FRONTEND_DIR"
npm run dev < /dev/null > "$LOGDIR/vite.log" 2>&1 &
VITE_PID=$!
for i in $(seq 1 30); do
  grep -q "Local:" "$LOGDIR/vite.log" 2>/dev/null && break
  sleep 1
done

echo "=== starting playwright driver ==="
NODE_PATH="$FRONTEND_DIR/node_modules" node "$REPO_ROOT/scripts/drive_capture.cjs" \
  < /dev/null > "$LOGDIR/driver.log" 2>&1 &
DRIVER_PID=$!
sleep 2

elapsed=0
while true; do
  n=$(ls "$CAPTURE_DIR/frames" 2>/dev/null | wc -l | tr -d ' ')
  echo "frames: $n (elapsed ${elapsed}s)"
  if [ "$n" -ge "$TARGET" ]; then
    echo "reached target ($TARGET)"
    break
  fi
  if [ "$elapsed" -ge "$MAXWAIT" ]; then
    echo "TIMEOUT waiting for frames after ${MAXWAIT}s, proceeding with what we have: $n"
    break
  fi
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "BACKEND DIED EARLY, elapsed ${elapsed}s, frames $n"
    cat "$LOGDIR/backend.log"
    break
  fi
  sleep 5
  elapsed=$((elapsed+5))
done

echo "=== stopping backend with kill -INT first (TERM leaves labels.json stale; INT reaches finalize()) ==="
kill -INT "$BACKEND_PID" 2>/dev/null
for i in $(seq 1 20); do
  kill -0 "$BACKEND_PID" 2>/dev/null || break
  sleep 1
done
if kill -0 "$BACKEND_PID" 2>/dev/null; then
  echo "backend did not exit after SIGINT + 20s, killing harder"
  kill -TERM "$BACKEND_PID" 2>/dev/null
fi

echo "=== cleaning up driver + vite ==="
kill "$DRIVER_PID" 2>/dev/null
kill "$VITE_PID" 2>/dev/null
sleep 1
pkill -f "drive_capture" 2>/dev/null
pkill -f "$FRONTEND_DIR/node_modules/.bin/vite" 2>/dev/null

DISK_FRAMES=$(ls "$CAPTURE_DIR/frames" 2>/dev/null | wc -l | tr -d ' ')
LABEL_IMAGES=$(python3 -c "import json; print(len(json.load(open('$CAPTURE_DIR/labels.json'))['images']))" 2>/dev/null || echo -1)

echo "=== final state ==="
echo "frames on disk: $DISK_FRAMES"
echo "labels.json images: $LABEL_IMAGES"

if [ "$DISK_FRAMES" != "$LABEL_IMAGES" ]; then
  echo "FATAL: labels.json image count ($LABEL_IMAGES) does not match frames on disk ($DISK_FRAMES)"
  exit 1
fi

echo "OK: labels.json matches frames on disk ($DISK_FRAMES)"
exit 0
