#!/bin/bash
# Service Startup Script for Ktiseos-Nyx-Trainer (RunPod)
# Use this to (re)start services WITHOUT re-provisioning.
#
# First time? Run provision_runpod.sh instead — it installs everything then starts services.
#
# Usage (from Jupyter terminal):
#   cd /workspace/Ktiseos-Nyx-Trainer && bash start_services_runpod.sh
#
# Access URLs (once running):
#   Frontend: https://{POD_ID}-3000.proxy.runpod.net
#   Backend:  https://{POD_ID}-8000.proxy.runpod.net
#   Jupyter:  https://{POD_ID}-8888.proxy.runpod.net (always available)

set -e

echo "=========================================="
echo "  Starting Ktiseos-Nyx-Trainer (RunPod)"
echo "=========================================="

# Activate virtual environment
for venv_path in "/venv/main/bin/activate" "/venv/bin/activate" "/workspace/venv/bin/activate"; do
    if [ -f "$venv_path" ]; then
        # shellcheck disable=SC1090
        source "$venv_path"
        echo "  Activated venv: $venv_path"
        break
    fi
done

# Ensure node is in PATH
if ! command -v node &> /dev/null; then
    export NVM_DIR="$HOME/.nvm"
    # shellcheck disable=SC1091
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

    for node_path in /opt/nvm/versions/node/*/bin /usr/bin /usr/local/bin; do
        if [ -f "$node_path/node" ]; then
            export PATH="$node_path:$PATH"
            break
        fi
    done
fi

# RunPod: Direct port binding (no Caddy proxy)
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

# Kill any existing services on our ports
echo "  Stopping any existing services..."
pkill -f "uvicorn api.main:app" 2>/dev/null || true
pkill -f "next-server" 2>/dev/null || true
pkill -f "node server.js" 2>/dev/null || true
sleep 1

# Verify environment
echo "  Verifying Python environment..."
if [ -f "installer.py" ]; then
    python installer.py --skip-install 2>/dev/null || echo "   (installer check skipped)"
fi

# Create log directory
mkdir -p /workspace/logs

# Start backend
if [ -d "api" ]; then
    echo "  Starting FastAPI backend on port $BACKEND_PORT..."
    python -m uvicorn api.main:app --host 0.0.0.0 --port "$BACKEND_PORT" 2>&1 | tee -a /workspace/logs/backend.log &
    BACKEND_PID=$!
    echo "   Backend PID: $BACKEND_PID"
else
    echo "  API directory not found - skipping backend"
fi

sleep 2

# Start frontend
if [ -d "frontend" ] && command -v node &> /dev/null; then
    if [ -d "frontend/.next" ]; then
        echo "  Starting Next.js frontend on port $FRONTEND_PORT..."
        cd frontend
        PORT=$FRONTEND_PORT BACKEND_PORT=$BACKEND_PORT NODE_ENV=production node server.js 2>&1 | tee -a /workspace/logs/frontend.log &
        FRONTEND_PID=$!
        echo "   Frontend PID: $FRONTEND_PID"
        cd ..
    else
        echo "  No frontend build found. Run provision_runpod.sh first, or:"
        echo "   python install_frontend.py"
    fi
else
    echo "  Frontend not available (missing directory or Node.js)"
fi

echo ""
echo "=========================================="
echo "  Services Running!"
echo "=========================================="
echo ""
if [ -n "$RUNPOD_POD_ID" ]; then
    echo "  Frontend: https://${RUNPOD_POD_ID}-${FRONTEND_PORT}.proxy.runpod.net"
    echo "  Backend:  https://${RUNPOD_POD_ID}-${BACKEND_PORT}.proxy.runpod.net"
    echo "  API Docs: https://${RUNPOD_POD_ID}-${BACKEND_PORT}.proxy.runpod.net/docs"
else
    echo "  Frontend: http://0.0.0.0:${FRONTEND_PORT}"
    echo "  Backend:  http://0.0.0.0:${BACKEND_PORT}"
fi
echo ""
echo "  Logs: /workspace/logs/backend.log, /workspace/logs/frontend.log"
echo ""
echo "  To stop: pkill -f uvicorn; pkill -f 'node server.js'"
echo ""

# Keep script alive
wait
