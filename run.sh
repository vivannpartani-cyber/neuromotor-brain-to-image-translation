#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  neuromotor — one-click launcher
#  Works on macOS, Linux, and Windows (Git Bash)
# ─────────────────────────────────────────────────────────────

# Find Python 3 regardless of OS / alias
if command -v python3 &>/dev/null; then
    PY=python3
    PIP=pip3
elif command -v python &>/dev/null; then
    PY=python
    PIP=pip
else
    echo "❌  Python not found. Please install Python 3.10+ from https://python.org"
    exit 1
fi

VERSION=$($PY -c "import sys; print(sys.version_info.minor)")
if [ "$VERSION" -lt 9 ]; then
    echo "❌  Python 3.9+ is required. You have $($PY --version)"
    exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🧠  NEUROMOTOR — Brainwaves → Images                      ║"
echo "║      Stanford AIMI Proof-of-Concept                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "   Python found: $($PY --version)"
echo ""

# Install dependencies if needed
echo "▶  Checking dependencies ..."
$PIP install -q -r requirements.txt
echo "   ✓  Dependencies ready"
echo ""

# Run the CLI demo
$PY neuromotor_cli.py "$@"
