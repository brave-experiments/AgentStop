#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing glances (editable)..."
python -m pip install -e "$ROOT_DIR/glances-4.3.1"

echo "Installing agentstop (editable)..."
python -m pip install -e "$ROOT_DIR"

echo "Done."