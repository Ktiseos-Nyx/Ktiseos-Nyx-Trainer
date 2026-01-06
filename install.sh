#!/bin/bash
# Ktiseos-Nyx-Trainer Installation Script for Linux & macOS
# Verifies Python version and installs dependencies into a local venv.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Configuration ---
PYTHON_CMD="python3"
MIN_PYTHON_VERSION="3.10"
VENV_DIR=".venv"
INSTALLER_PY="installer_local_linux.py" # Assuming you'll create this from the windows one

echo "=========================================="
echo "Ktiseos-Nyx-Trainer Installation"
echo "=========================================="
echo ""

# --- OS Check ---
OS="$(uname -s)"
if [[ "$OS" != "Linux" && "$OS" != "Darwin" ]]; then
    echo "❌ This script is for Linux or macOS only."
    echo "   On Windows, please double-click 'install_windows_local.bat' instead."
    exit 1
fi

# --- Python Check ---
# 1. Check if python3 command exists
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "❌ Error: '$PYTHON_CMD' command not found."
    echo "   Please install Python ${MIN_PYTHON_VERSION} or newer and ensure '$PYTHON_CMD' is in your PATH."
    exit 1
fi

# 2. Check if the found Python is new enough
PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
MIN_VERSION_COMPARE=$(printf '%s\n' "$MIN_PYTHON_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)

echo "🐍 Found Python version: $PYTHON_VERSION"
if [ "$MIN_VERSION_COMPARE" != "$MIN_PYTHON_VERSION" ]; then
    echo "❌ Error: Python version $PYTHON_VERSION is too old."
    echo "   This application requires Python ${MIN_PYTHON_VERSION} or newer."
    exit 1
fi

# --- Virtual Environment and Installation ---
# 3. Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "🌍 Creating Python virtual environment in './$VENV_DIR'..."
    $PYTHON_CMD -m venv "$VENV_DIR"
else
    echo "✅ Virtual environment already exists."
fi

# 4. Activate venv and run the Python installer script
echo "📦 Activating environment and running the installer script..."
source "$VENV_DIR/bin/activate"

# Now that the venv is active, 'python' refers to the one inside the venv
python "$INSTALLER_PY" "$@"

# The 'deactivate' command is not strictly necessary as the script will exit,
# but it's good practice if you were to add more commands here.
deactivate

echo ""
echo "✅ Installation script finished!"
echo "   Dependencies are installed in the local './${VENV_DIR}' directory."
echo ""
echo "Next steps:"
echo "  - To start the application, run: ./start_services_local.sh"
echo ""