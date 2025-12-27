#!/bin/bash
# Ktiseos-Nyx-Trainer Installation Script
# Auto-selects the correct installer based on OS

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Ktiseos-Nyx-Trainer Installation"
echo "=========================================="
echo ""

# Detect OS
OS="$(uname -s)"
case "${OS}" in
  Linux*)   PLATFORM="linux" ;;
  Darwin*)  PLATFORM="macos" ;;
  CYGWIN*|MINGW*|MSYS*)
    echo "❌ This script is for Linux/macOS only."
    echo "   On Windows, use install.bat instead."
    exit 1
    ;;
  *)        PLATFORM="unknown" ;;
esac

# Select installer
if [[ "$PLATFORM" == "linux" || "$PLATFORM" == "macos" ]]; then
  if [ -f "installer_local_linux.py" ]; then
    INSTALLER="installer_local_linux.py"
  else
    echo "❌ installer_local_linux.py not found!"
    exit 1
  fi
fi

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
echo "📦 Using installer: $INSTALLER"
echo ""

# Run installer with all args
"$PYTHON_CMD" "$INSTALLER" "$@"

echo ""
echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "  - Local development: ./start_services_local.sh"
echo "  - Windows users: Use install.bat and start_services_local.bat"
echo ""
