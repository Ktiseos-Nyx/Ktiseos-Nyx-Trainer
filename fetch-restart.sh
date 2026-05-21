#!/bin/bash
# Fetch-Restart Script for Ktiseos-Nyx-Trainer
# Full update: git pull + npm install + frontend build + service restart
# Use restart.sh instead for a quick restart without pulling/rebuilding.

set -e

echo "=========================================="
echo "🔄 Fetch-Restart: Pull → Install → Build → Start"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Auto-detect environment
if [ -d "/workspace" ]; then
    HOST="0.0.0.0"
    BACKEND_PORT="${BACKEND_PORT:-18000}"
    FRONTEND_PORT="${FRONTEND_PORT:-13000}"
    echo "🌐 Detected remote GPU environment (VastAI/RunPod)"
else
    HOST="127.0.0.1"
    BACKEND_PORT="8000"
    FRONTEND_PORT="3000"
    echo "💻 Detected local development environment"
fi

DISPLAY_HOST="localhost"

# ----------------------------------------------------------------
# Step 1: Git pull
# ----------------------------------------------------------------
echo ""
echo "📥 Pulling latest changes..."
git pull

# ----------------------------------------------------------------
# Step 2: Frontend install + build (only if frontend/ exists)
# ----------------------------------------------------------------
if [ -d "frontend" ]; then
    echo ""
    echo "📦 Installing frontend dependencies..."
    cd frontend
    npm install

    echo ""
    echo "🏗️  Building frontend..."
    npm run build
    cd "$SCRIPT_DIR"
else
    echo "⚠️  frontend/ not found — skipping npm steps"
fi

# ----------------------------------------------------------------
# Step 3: Restart services
# ----------------------------------------------------------------
echo ""
echo "=========================================="
echo "🚀 Restarting Services..."
echo "=========================================="

if [ -d "/workspace" ] && command -v supervisorctl &>/dev/null; then
    supervisorctl restart ktiseos-nyx

    echo "⏳ Waiting for services to come up..."
    sleep 10

    echo "🔄 Warming up Caddy connection..."
    curl -sf "http://localhost:${FRONTEND_PORT}/api/health" > /dev/null 2>&1 || true

    echo ""
    echo "=========================================="
    echo "✅ Fetch-Restart Complete!"
    echo "=========================================="
    echo "📡 Backend:  http://${DISPLAY_HOST}:${BACKEND_PORT}"
    echo "🎨 Frontend: http://${DISPLAY_HOST}:${FRONTEND_PORT}"
    echo "📊 Logs:     /workspace/logs/supervisor.log"
    echo "=========================================="
else
    # Local: kill by port, then start
    echo "🧹 Cleaning up existing processes..."
    lsof -ti:$BACKEND_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
    lsof -ti:$FRONTEND_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
    sleep 2

    echo "🔧 Starting FastAPI backend on http://${HOST}:${BACKEND_PORT}..."
    python -m uvicorn api.main:app --host "$HOST" --port "$BACKEND_PORT" --reload &
    BACKEND_PID=$!
    sleep 3

    echo "🎨 Starting Next.js frontend on http://${HOST}:${FRONTEND_PORT}..."
    cd frontend
    NODE_ENV=production PORT=$FRONTEND_PORT BACKEND_PORT=$BACKEND_PORT node server.js &
    FRONTEND_PID=$!
    cd "$SCRIPT_DIR"
    sleep 3

    echo ""
    echo "=========================================="
    echo "✅ Fetch-Restart Complete!"
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
