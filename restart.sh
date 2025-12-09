#!/bin/bash
# Quick Restart Script for Ktiseos-Nyx-Trainer
# Skips dependency installation for fast restarts

set -e

echo "=========================================="
echo "⚡ Quick Restart (Skipping Dependency Installation)..."
echo "=========================================="

# Clean up existing processes
echo "🧹 Cleaning up existing processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
pkill -f "uvicorn.*api.main" 2>/dev/null || true
pkill -f "node.*next" 2>/dev/null || true

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
echo "🔧 Starting FastAPI backend on http://127.0.0.1:8000..."
cd "$(dirname "$0")"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!

# Give backend time to start
sleep 3

# Start Next.js frontend
echo "🎨 Starting Next.js frontend on http://127.0.0.1:3000..."
cd frontend
npm run dev -- -H 127.0.0.1 &
FRONTEND_PID=$!

# Wait a moment for services to start
sleep 3

echo ""
echo "=========================================="
echo "✅ Services Started!"
echo "=========================================="
echo "📡 Backend:  http://127.0.0.1:8000"
echo "🎨 Frontend: http://127.0.0.1:3000"
echo "📊 API Docs: http://127.0.0.1:8000/docs"
echo ""
echo "Backend PID:  $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "Press Ctrl+C to stop all services"
echo "=========================================="

# Wait for Ctrl+C
trap "echo '⚠️ Stopping services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM
wait
