#!/usr/bin/env bash
# Build the packaged StreetLab.app: PyInstaller one-file sidecar, placed at
# the target-triple-suffixed path Tauri's `externalBin` resolution expects,
# then a real `tauri build`. Run from anywhere; paths are resolved relative
# to this script's location.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BACKEND_DIR="$ROOT/streetlab-backend"
FRONTEND_DIR="$ROOT/streetlab"
BIN_DIR="$FRONTEND_DIR/src-tauri/bin"

if ! command -v rustc >/dev/null 2>&1; then
  echo "error: rustc not found on PATH (needed to derive the target triple)" >&2
  exit 1
fi
TRIPLE="$(rustc -vV | awk '/^host/{print $2}')"
SIDECAR_NAME="streetlab-server-$TRIPLE"

echo "== 1/3: building the PyInstaller sidecar ($TRIPLE) =="
cd "$BACKEND_DIR"
rm -rf build dist
rm -f streetlab-server.spec
uv run pyinstaller --onefile --name streetlab-server server/cli.py

mkdir -p "$BIN_DIR"
cp "dist/streetlab-server" "$BIN_DIR/$SIDECAR_NAME"
chmod +x "$BIN_DIR/$SIDECAR_NAME"
SIDECAR_SIZE="$(du -h "$BIN_DIR/$SIDECAR_NAME" | cut -f1)"
echo "sidecar: $BIN_DIR/$SIDECAR_NAME ($SIDECAR_SIZE)"

echo "== 2/3: building the Tauri app =="
cd "$FRONTEND_DIR"
npm run tauri build

APP_PATH="$(find "$FRONTEND_DIR/src-tauri/target/release/bundle/macos" -maxdepth 1 -iname "*.app" | head -1)"
if [ -z "$APP_PATH" ]; then
  echo "error: no .app bundle found under src-tauri/target/release/bundle/macos" >&2
  exit 1
fi
APP_SIZE="$(du -sh "$APP_PATH" | cut -f1)"

echo "== 3/3: done =="
echo "sidecar binary : $SIDECAR_SIZE  ($BIN_DIR/$SIDECAR_NAME)"
echo ".app bundle    : $APP_SIZE  ($APP_PATH)"
