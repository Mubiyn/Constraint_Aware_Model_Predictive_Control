#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="python"

log() {
  printf "[%s] %s\n" "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$1"
}

log "Installing Python dependencies (editable install + dev extras)"
cd "$ROOT_DIR"
$PYTHON -m pip install --upgrade pip setuptools
$PYTHON -m pip install -e .[dev]

cat <<'EOF'
Dependencies installed. If you prefer to skip this step because the packages are already available in your active conda environment (e.g., drone-rl-pid), just skip this script and rely on that environment instead.
EOF
