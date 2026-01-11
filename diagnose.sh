#!/bin/bash

echo "======================================================================"
echo "Ktiseos-Nyx-Trainer Diagnostic Tool"
echo "======================================================================"
echo ""
echo "This tool will collect system information to help troubleshoot"
echo "installation issues. It will create two files:"
echo "  - diagnostics_TIMESTAMP.json (for developers)"
echo "  - diagnostics_TIMESTAMP.txt (human-readable report)"
echo ""
echo "======================================================================"
echo ""

# Find Python
PYTHON_CMD=""

if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "ERROR: Python not found!"
    echo ""
    echo "Please install Python 3.10+ using your package manager:"
    echo "  Ubuntu/Debian: sudo apt install python3"
    echo "  Fedora: sudo dnf install python3"
    echo "  Arch: sudo pacman -S python"
    echo ""
    exit 1
fi

echo "Using Python: $PYTHON_CMD"
echo ""

$PYTHON_CMD diagnose.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Diagnostic script failed."
    echo ""
    exit 1
fi

echo ""
echo "======================================================================"
echo "Diagnostic files created successfully!"
echo "======================================================================"
echo ""
echo "Please attach the diagnostics_*.txt file to your GitHub issue."
echo ""
