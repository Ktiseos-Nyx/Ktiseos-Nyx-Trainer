#!/bin/bash
# Service Startup Script for Ktiseos-Nyx-Trainer (Local Unix)
# This script activates the virtual environment and starts the backend/frontend.
#
# Usage: ./start_services_local.sh [--port 3000] [--backend-port 8000]
#        ./start_services_local.sh -h | --help

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
VENV_DIR=".venv"
HAS_WARNINGS=0
FRONTEND_PORT=3000
BACKEND_PORT=8000

# --- Parse arguments ---
validate_port() {
    local flag="$1" val="$2"
    if [ -z "$val" ] || [ "${val#-}" != "$val" ]; then
        echo "[ERROR] $flag requires a port number."
        exit 1
    fi
    case "$val" in
        *[!0-9]*) echo "[ERROR] $flag requires a numeric port (got: $val)."; exit 1 ;;
    esac
    if [ "$val" -lt 1 ] || [ "$val" -gt 65535 ]; then
        echo "[ERROR] $flag port must be between 1 and 65535 (got: $val)."
        exit 1
    fi
}

while [ $# -gt 0 ]; do
    case "$1" in
        --port)
            validate_port "--port" "$2"
            FRONTEND_PORT="$2"; shift 2 ;;
        --backend-port)
            validate_port "--backend-port" "$2"
            BACKEND_PORT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--port FRONTEND_PORT] [--backend-port BACKEND_PORT]"
            echo ""
            echo "  --port           Frontend port (default: 3000)"
            echo "  --backend-port   Backend API port (default: 8000)"
            echo ""
            echo "Example: $0 --port 4000 --backend-port 9000"
            exit 0 ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            echo "Run '$0 --help' for usage."
            exit 1 ;;
    esac
done

# --- Main Script ---
echo "=========================================="
echo "Starting Ktiseos-Nyx-Trainer Services (Local)..."
echo "=========================================="
echo ""

# ====================================================================
#  PRE-FLIGHT CHECKS - Catch common issues BEFORE they become cryptic
# ====================================================================
echo "[Pre-flight checks...]"
echo ""

# --- Check: File ownership mismatch (root vs user) ---
# This is the #1 cause of "Permission denied" on Linux.
# Happens when: cloned as root, running as user (or vice versa)
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -d "$PROJECT_DIR/frontend" ]; then
    DIR_OWNER=$(stat -c '%u' "$PROJECT_DIR/frontend" 2>/dev/null || stat -f '%u' "$PROJECT_DIR/frontend" 2>/dev/null || echo "unknown")
    CURRENT_UID=$(id -u)
    if [ "$DIR_OWNER" != "unknown" ] && [ "$DIR_OWNER" != "$CURRENT_UID" ]; then
        OWNER_NAME=$(id -un "$DIR_OWNER" 2>/dev/null || echo "uid:$DIR_OWNER")
        echo "[WARNING] File ownership mismatch!"
        echo "          Project files are owned by: $OWNER_NAME (uid:$DIR_OWNER)"
        echo "          You are running as: $(whoami) (uid:$CURRENT_UID)"
        echo ""
        echo "          This WILL cause 'Permission denied' errors."
        echo "          Fix: sudo chown -R $(whoami):$(id -gn) $PROJECT_DIR"
        echo ""
        HAS_WARNINGS=1
    fi
fi

# --- Check: Write permissions ---
if ! touch "$PROJECT_DIR/.write_test" 2>/dev/null; then
    echo "[ERROR] Cannot write to project folder!"
    echo "        Path: $PROJECT_DIR"
    echo ""
    echo "        Fix: sudo chown -R $(whoami):$(id -gn) $PROJECT_DIR"
    echo "        Or:  chmod -R u+w $PROJECT_DIR"
    echo ""
    exit 1
else
    rm -f "$PROJECT_DIR/.write_test"
fi

# --- Check: node_modules ownership (common after sudo npm install) ---
if [ -d "$PROJECT_DIR/frontend/node_modules" ]; then
    NM_OWNER=$(stat -c '%u' "$PROJECT_DIR/frontend/node_modules" 2>/dev/null || stat -f '%u' "$PROJECT_DIR/frontend/node_modules" 2>/dev/null || echo "unknown")
    if [ "$NM_OWNER" != "unknown" ] && [ "$NM_OWNER" != "$(id -u)" ]; then
        echo "[WARNING] frontend/node_modules is owned by a different user!"
        echo "          This happens if 'npm install' was run as root/sudo."
        echo "          Fix: sudo chown -R $(whoami):$(id -gn) $PROJECT_DIR/frontend/node_modules"
        echo "          Or delete and reinstall: rm -rf frontend/node_modules && cd frontend && npm install"
        echo ""
        HAS_WARNINGS=1
    fi
fi

# --- Check: Ports already in use ---
if command -v ss &> /dev/null; then
    PORT_CHECK_CMD="ss -tlnp"
elif command -v netstat &> /dev/null; then
    PORT_CHECK_CMD="netstat -tlnp"
else
    PORT_CHECK_CMD=""
fi

if [ -n "$PORT_CHECK_CMD" ]; then
    if $PORT_CHECK_CMD 2>/dev/null | grep -q ":${BACKEND_PORT} "; then
        echo "[WARNING] Port ${BACKEND_PORT} is already in use!"
        echo "          Something is already listening there. The backend may fail."
        echo "          Check with: $PORT_CHECK_CMD 2>/dev/null | grep :${BACKEND_PORT}"
        echo "          Tip: use --backend-port to pick a different port."
        echo ""
        HAS_WARNINGS=1
    fi
    if $PORT_CHECK_CMD 2>/dev/null | grep -q ":${FRONTEND_PORT} "; then
        echo "[WARNING] Port ${FRONTEND_PORT} is already in use!"
        echo "          Something is already listening there. The frontend may fail."
        echo "          Check with: $PORT_CHECK_CMD 2>/dev/null | grep :${FRONTEND_PORT}"
        echo "          Tip: use --port to pick a different port."
        echo ""
        HAS_WARNINGS=1
    fi
fi

# --- Check: Stale node processes ---
if pgrep -f "node.*server.js" > /dev/null 2>&1 || pgrep -f "next-server" > /dev/null 2>&1; then
    echo "[INFO] Existing Node.js processes found (possibly from a previous run)."
    echo "       If the frontend fails, try: pkill -f 'node.*server.js'"
    echo ""
fi

if [ "$HAS_WARNINGS" = "1" ]; then
    echo "------------------------------------------"
    echo "  Warnings found. Services may still work,"
    echo "  but check above if you hit errors."
    echo "------------------------------------------"
    echo ""
fi

echo "[Pre-flight checks complete.]"
echo ""

# ====================================================================
# Step 1: Verify that the environment has been installed
# ====================================================================
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

# ====================================================================
# Step 2: Start Services
# ====================================================================

# Trap SIGINT (Ctrl+C) to gracefully shut down background processes
trap 'echo "\n[INFO] Shutting down services..."; pkill -P $$; exit' SIGINT

# Start FastAPI backend in the background
if [ -d "api" ]; then
    echo "[Backend] Starting FastAPI backend on http://localhost:${BACKEND_PORT}..."
    $PYTHON_CMD -m uvicorn api.main:app --host 127.0.0.1 --port "$BACKEND_PORT" --reload &
else
    echo "[Warning] API directory not found - skipping backend startup."
fi

# Start Next.js frontend in the background
if [ -d "frontend" ]; then
    # Check if npm is available
    if ! command -v npm &> /dev/null; then
        echo "[Warning] npm not found - skipping frontend startup."
        echo "          Install Node.js 20+ to enable frontend."
    else
        # Auto-install deps if missing
        if [ ! -d "frontend/node_modules/next" ]; then
            echo "[Frontend] Dependencies missing, running npm install..."
            (cd frontend && npm install --legacy-peer-deps)
        fi
        # Auto-build if missing
        if [ ! -d "frontend/.next" ]; then
            echo "[Frontend] No build found, running npm run build..."
            (cd frontend && npm run build)
        fi
        echo "[Frontend] Starting Next.js frontend on http://localhost:${FRONTEND_PORT}..."
        (cd frontend && NODE_ENV=production PORT="$FRONTEND_PORT" BACKEND_PORT="$BACKEND_PORT" npm start) &
    fi
else
    echo "[Warning] Frontend directory not found - skipping frontend startup."
fi

echo ""
echo "=========================================="
echo "[SUCCESS] Local Services Started!"
echo "=========================================="
echo ""
echo ">> Access the UI at: http://localhost:${FRONTEND_PORT}"
echo ">> API Docs available at: http://localhost:${BACKEND_PORT}/docs"
echo ""
echo "[INFO] Services are running in the background."
echo "[INFO] Press CTRL+C in this terminal to stop all services."
echo ""

# Wait for all background jobs to finish. The 'trap' will handle shutdown.
wait
