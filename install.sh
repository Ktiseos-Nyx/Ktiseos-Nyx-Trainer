#!/bin/bash
# Ktiseos-Nyx-Trainer Installation Script
# Simple wrapper that calls installer.py with the current Python environment

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Ktiseos-Nyx-Trainer Installation"
echo "=========================================="
echo ""

# Find Python 3
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "❌ Error: Python 3 not found!"
    echo "   Please install Python 3.10+ and try again."
    exit 1
fi

echo "🐍 Using Python: $PYTHON_CMD ($($PYTHON_CMD --version))"
echo ""

# Run the installer with all arguments passed through
$PYTHON_CMD installer.py "$@"

echo ""
echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "  - Local development: ./start_services_local.sh"
echo "  - VastAI deployment: Services auto-start via supervisor"
echo ""
