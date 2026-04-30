#!/bin/bash
# Quick Restart Script for Ktiseos-Nyx-Trainer
# Skips dependency installation for fast restarts

set -e

echo "=========================================="
echo "⚡ Quick Restart (Skipping Dependency Installation)..."
echo "=========================================="

# Auto-detect environment: if /workspace exists, we're on a remote GPU (VastAI/RunPod)
if [ -d "/workspace" ]; then
    HOST="0.0.0.0"
    BACKEND_PORT="${BACKEND_PORT:-8000}"
    FRONTEND_PORT="${FRONTEND_PORT:-3000}"
    echo "🌐 Detected remote GPU environment (binding to 0.0.0.0)"
else
    HOST="127.0.0.1"
    BACKEND_PORT="8000"
    FRONTEND_PORT="3000"
    echo "💻 Detected local development environment"
fi

DISPLAY_HOST="${HOST}"
[ "$HOST" = "0.0.0.0" ] && DISPLAY_HOST="localhost"

if [ -d "/workspace" ] && command -v supervisorctl &>/dev/null; then
    # ----------------------------------------------------------------
    # VastAI / RunPod: let supervisor own the full restart
    # Avoids broad pkill patterns that can kill VastAI's portal processes
    # ----------------------------------------------------------------
    echo "🔄 Restarting via supervisorctl (ktiseos-nyx)..."
    supervisorctl restart ktiseos-nyx

    echo "⏳ Waiting for services to come up..."
    sleep 10

    # Warm up the connection so Caddy re-establishes its backend proxy
    echo "🔄 Warming up Caddy connection..."
    curl -sf "http://localhost:${FRONTEND_PORT}/api/health" > /dev/null 2>&1 || true

    echo ""
    echo "=========================================="
    echo "✅ Services Restarted!"
    echo "=========================================="
    echo "📡 Backend:  http://${DISPLAY_HOST}:${BACKEND_PORT}"
    echo "🎨 Frontend: http://${DISPLAY_HOST}:${FRONTEND_PORT}"
    echo "📊 Logs:     /workspace/logs/supervisor.log"
    echo "=========================================="
else
    # ----------------------------------------------------------------
    # Local: kill by port only, then start manually
    # ----------------------------------------------------------------
    echo "🧹 Cleaning up existing processes..."
    lsof -ti:$BACKEND_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
    lsof -ti:$FRONTEND_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
    sleep 2

    echo "⚙️ Verifying environment (skipping reinstall)..."
    python installer.py --skip-install

    echo ""
    echo "=========================================="
    echo "🚀 Starting Services..."
    echo "=========================================="

    echo "🔧 Starting FastAPI backend on http://${HOST}:${BACKEND_PORT}..."
    cd "$(dirname "$0")"
    python -m uvicorn api.main:app --host "$HOST" --port "$BACKEND_PORT" --reload &
    BACKEND_PID=$!
    sleep 3

    echo "🎨 Starting Next.js frontend on http://${HOST}:${FRONTEND_PORT}..."
    cd frontend
    NODE_ENV=production PORT=$FRONTEND_PORT BACKEND_PORT=$BACKEND_PORT npm start &
    FRONTEND_PID=$!
    sleep 3

    echo ""
    echo "=========================================="
    echo "✅ Services Started!"
    echo "=========================================="
    echo "📡 Backend:  http://${DISPLAY_HOST}:${BACKEND_PORT}"
    echo "🎨 Frontend: http://${DISPLAY_HOST}:${FRONTEND_PORT}"
    echo "📊 API Docs: http://${DISPLAY_HOST}:${BACKEND_PORT}/docs"
    echo ""
    echo "Backend PID:  $BACKEND_PID"
    echo "Frontend PID: $FRONTEND_PID"
    echo ""
    echo "Press Ctrl+C to stop all services"
    echo "=========================================="

    trap "echo '⚠️ Stopping services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM
    wait
fi
