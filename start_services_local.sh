#!/bin/bash
# Service Startup Script for Ktiseos-Nyx-Trainer (Local Unix)
# This script activates the virtual environment and starts the backend/frontend.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
VENV_DIR=".venv"

# --- Main Script ---
echo "=========================================="
echo "Starting Ktiseos-Nyx-Trainer Services (Local)..."
echo "=========================================="
echo ""

# --------------------------------------------------------------------
# Step 1: Verify that the environment has been installed
# --------------------------------------------------------------------
echo "[Verifying installation...]"
if [ -d "$VENV_DIR/bin" ]; then
    echo "[OK] Virtual environment found. Activating..."
    source "$VENV_DIR/bin/activate"
    PYTHON_CMD="python"
    echo "   Python executable: $(which python)"
else
    echo "[OK] No virtual environment found. Using system Python."
    # Verify system Python is available
    if ! command -v python3 &> /dev/null; then
        echo "[ERROR] Python 3 not found!"
        echo "   Please run './install.sh' to set up the environment first."
        echo ""
        exit 1
    fi
    PYTHON_CMD="python3"
    echo "   Python executable: $(which python3)"
fi
echo ""

# --------------------------------------------------------------------
# Step 3: Clean up any existing processes
# --------------------------------------------------------------------
echo "[Cleanup] Stopping any existing services on ports 8000 & 3000..."
# Use pkill for simplicity and robustness. The '|| true' prevents script exit if no process is found.
pkill -f "uvicorn api.main:app" || true
pkill -f "npm.*start" || true
sleep 1

# --------------------------------------------------------------------
# Step 4: Start Services
# --------------------------------------------------------------------

# Trap SIGINT (Ctrl+C) to gracefully shut down background processes
trap 'echo "\n[INFO] Shutting down services..."; pkill -P $$; exit' SIGINT

# Start FastAPI backend in the background
if [ -d "api" ]; then
    echo "[Backend] Starting FastAPI backend on http://localhost:8000..."
    $PYTHON_CMD -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload &
else
    echo "[Warning] API directory not found - skipping backend startup."
fi

# Start Next.js frontend in the background
if [ -d "frontend" ]; then
    # Check if npm is available
    if ! command -v npm &> /dev/null; then
        echo "[Warning] npm not found - skipping frontend startup."
        echo "          Install Node.js 18+ to enable frontend."
    else
        echo "[Frontend] Starting Next.js frontend on http://localhost:3000..."
        (cd frontend && npm start &)
    fi
else
    echo "[Warning] Frontend directory not found - skipping frontend startup."
fi

echo ""
echo "=========================================="
echo "[SUCCESS] Local Services Started!"
echo "=========================================="
echo ""
echo ">> Access the UI at: http://localhost:3000"
echo ">> API Docs available at: http://localhost:8000/docs"
echo ""
echo "[INFO] Services are running in the background."
echo "[INFO] Press CTRL+C in this terminal to stop all services."
echo ""

# Wait for all background jobs to finish. The 'trap' will handle shutdown.
wait