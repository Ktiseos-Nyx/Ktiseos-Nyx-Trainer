#!/bin/bash
# Ktiseos-Nyx-Trainer Installation Script for Linux & macOS
# Supports interactive venv creation and flag-based automation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Configuration ---
PYTHON_CMD="python3"
MIN_PYTHON_VERSION="3.10"
VENV_DIR=".venv"
INSTALLER_PY="installer.py"
USE_VENV=""
AUTO_MODE=0

# --- Parse Arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --venv)
            USE_VENV=1
            shift
            ;;
        --no-venv)
            USE_VENV=0
            shift
            ;;
        --auto)
            USE_VENV=1
            AUTO_MODE=1
            shift
            ;;
        *)
            # Keep other arguments to pass to installer
            INSTALLER_ARGS="$INSTALLER_ARGS $1"
            shift
            ;;
    esac
done

echo "=========================================="
echo "Ktiseos-Nyx-Trainer Installation"
echo "=========================================="
echo ""

# --- OS Check ---
OS="$(uname -s)"
if [[ "$OS" != "Linux" && "$OS" != "Darwin" ]]; then
    echo "❌ This script is for Linux or macOS only."
    echo "   On Windows, please use 'install.bat' instead."
    exit 1
fi

# --- Python Check ---
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "❌ Error: '$PYTHON_CMD' command not found."
    echo "   Please install Python ${MIN_PYTHON_VERSION} or newer."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
MIN_VERSION_COMPARE=$(printf '%s\n' "$MIN_PYTHON_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)

echo "🐍 Found Python version: $PYTHON_VERSION"
if [ "$MIN_VERSION_COMPARE" != "$MIN_PYTHON_VERSION" ]; then
    echo "❌ Error: Python version $PYTHON_VERSION is too old."
    echo "   This application requires Python ${MIN_PYTHON_VERSION} or newer."
    exit 1
fi
echo ""

# --- Pre-flight: Permission & Environment Checks ---
echo "[Pre-flight checks...]"

# Check write permissions
if ! touch "$SCRIPT_DIR/.write_test" 2>/dev/null; then
    echo "❌ ERROR: Cannot write to project folder!"
    echo "   Path: $SCRIPT_DIR"
    echo ""
    echo "   This usually means file ownership doesn't match your user."
    echo "   Fix: sudo chown -R $(whoami):$(id -gn) $SCRIPT_DIR"
    echo ""
    exit 1
else
    rm -f "$SCRIPT_DIR/.write_test"
fi

# Check ownership mismatch (most common Linux "Permission denied")
DIR_OWNER=$(stat -c '%u' "$SCRIPT_DIR" 2>/dev/null || stat -f '%u' "$SCRIPT_DIR" 2>/dev/null || echo "unknown")
CURRENT_UID=$(id -u)
if [ "$DIR_OWNER" != "unknown" ] && [ "$DIR_OWNER" != "$CURRENT_UID" ]; then
    OWNER_NAME=$(id -un "$DIR_OWNER" 2>/dev/null || echo "uid:$DIR_OWNER")
    echo "⚠️  WARNING: File ownership mismatch!"
    echo "   Project files owned by: $OWNER_NAME (uid:$DIR_OWNER)"
    echo "   You are running as: $(whoami) (uid:$CURRENT_UID)"
    echo ""
    echo "   This may cause 'Permission denied' errors during install."
    echo "   Fix: sudo chown -R $(whoami):$(id -gn) $SCRIPT_DIR"
    echo ""
fi

# Warn against running installer as root (unless on VastAI/Docker)
if [ "$CURRENT_UID" = "0" ] && [ ! -f "/.dockerenv" ] && [ ! -d "/workspace" ]; then
    echo "⚠️  WARNING: Running as root on a non-container system."
    echo "   This will create files owned by root that your normal user can't modify."
    echo "   Consider running as your regular user instead."
    echo ""
fi

echo "[Pre-flight checks complete.]"
echo ""

# --- Interactive venv prompt if not specified ---
if [ -z "$USE_VENV" ]; then
    echo "Virtual Environment Recommendation:"
    echo "=========================================="
    echo "A virtual environment (venv) isolates this project's packages from"
    echo "other Python projects on your system. This prevents version conflicts"
    echo "and makes troubleshooting much easier."
    echo ""
    echo "Benefits:"
    echo "  - No conflicts with other Python projects"
    echo "  - Easy to delete if you want to start fresh (just 'rm -rf .venv')"
    echo "  - Industry best practice for Python development"
    echo ""
    echo "The venv will be created in: $SCRIPT_DIR/$VENV_DIR"
    echo "=========================================="
    echo ""
    read -p "Create a virtual environment? (Y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        USE_VENV=0
        echo ""
        echo "Proceeding WITHOUT virtual environment..."
        echo "(You can always re-run with --venv flag later)"
        echo ""
    else
        USE_VENV=1
        echo ""
        echo "Creating virtual environment..."
        echo ""
    fi
fi

# --- Create and use venv if requested ---
if [ "$USE_VENV" = "1" ]; then
    if [ ! -d "$VENV_DIR/bin/activate" ]; then
        echo "Creating virtual environment in '$VENV_DIR'..."
        $PYTHON_CMD -m venv "$VENV_DIR"
        if [ $? -ne 0 ]; then
            echo ""
            echo "❌ ERROR: Failed to create virtual environment."
            echo "   Make sure Python 3.10+ is installed."
            exit 1
        fi
        echo "Virtual environment created successfully!"
        echo ""
    else
        echo "Virtual environment already exists."
        echo ""
    fi

    echo "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    echo ""
    echo "(venv) Virtual environment activated."
    echo "Python: $VENV_DIR/bin/python"
    echo ""
    echo "Running installer inside venv..."
    echo ""

    # Run installer with venv Python
    python "$INSTALLER_PY" $INSTALLER_ARGS

    deactivate

    echo ""
    echo "✅ Installation complete!"
    echo "   Dependencies are installed in: ./$VENV_DIR"
    echo ""
    echo "Next steps:"
    echo "  - Activate venv: source $VENV_DIR/bin/activate"
    echo "  - Start services: ./start_services_local.sh"
    echo ""
else
    echo "Running installer WITHOUT virtual environment..."
    echo ""

    # Run installer directly
    $PYTHON_CMD "$INSTALLER_PY" $INSTALLER_ARGS

    echo ""
    echo "✅ Installation complete!"
    echo ""
    echo "Next steps:"
    echo "  - Start services: ./start_services_local.sh"
    echo ""
fi