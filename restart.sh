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

# Clean up existing processes — graceful SIGTERM first, SIGKILL only as fallback
# (kill -9 immediately confuses VastAI's Caddy proxy by yanking the connection)
echo "🧹 Cleaning up existing processes..."
pkill -TERM -f "uvicorn.*api.main" 2>/dev/null || true
pkill -TERM -f "node.*next" 2>/dev/null || true
sleep 3
# Force-kill anything still alive after graceful window
lsof -ti:$BACKEND_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
lsof -ti:$FRONTEND_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 2

# Run installer with --skip-install flag
echo "⚙️ Verifying environment (skipping reinstall)..."
python installer.py --skip-install

# Start services
echo ""
echo "=========================================="
echo "🚀 Starting Services..."
echo "=========================================="

# Start FastAPI backend
echo "🔧 Starting FastAPI backend on http://${HOST}:${BACKEND_PORT}..."
cd "$(dirname "$0")"
python -m uvicorn api.main:app --host "$HOST" --port "$BACKEND_PORT" --reload &
BACKEND_PID=$!

# Give backend time to start
sleep 3

# Start Next.js frontend
# Export BACKEND_PORT so upload-zip route (which reads process.env.BACKEND_PORT) can find FastAPI
echo "🎨 Starting Next.js frontend on http://${HOST}:${FRONTEND_PORT}..."
cd frontend
NODE_ENV=production PORT=$FRONTEND_PORT BACKEND_PORT=$BACKEND_PORT npm start &
FRONTEND_PID=$!

# Wait for services to start
sleep 5

# On VastAI: make a health request so Caddy re-establishes its backend connection
if [ -d "/workspace" ]; then
    echo "🔄 Warming up Caddy connection..."
    sleep 3
    curl -sf "http://localhost:${FRONTEND_PORT}/api/health" > /dev/null 2>&1 || true
fi

DISPLAY_HOST="${HOST}"
[ "$HOST" = "0.0.0.0" ] && DISPLAY_HOST="localhost"

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

# Wait for Ctrl+C
trap "echo '⚠️ Stopping services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM
wait
