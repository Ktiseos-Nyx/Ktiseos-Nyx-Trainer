#!/bin/bash
# Service Startup Script for Ktiseos-Nyx-Trainer (Vast.ai / Cloud Deployment)
# This script starts both the FastAPI backend and Next.js frontend, binding to 0.0.0.0

set -e

echo "=========================================="
echo "🚀 Starting Ktiseos-Nyx-Trainer Services (Vast.ai/Cloud)..."
echo "=========================================="

# --------------------------------------------------------------------
# Step 0: Clean up any existing processes using the ports
# --------------------------------------------------------------------
echo "🧹 Cleaning up any existing processes on ports 8000 and 3000..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

# Also kill any existing uvicorn or next processes to prevent conflicts
pkill -f "uvicorn.*api.main" 2>/dev/null || true
pkill -f "node.*next" 2>/dev/null || true

sleep 3  # Give time for processes to terminate

# --------------------------------------------------------------------
# Step 1: Ensure environment is set up via installer.py
# Note: VastAI PyTorch template might already have venv activated.
# The installer.py will verify and install missing components.
# --------------------------------------------------------------------
echo "⚙️ Running unified installer.py to set up environment..."
python installer.py

# --------------------------------------------------------------------
# Step 2: Start Services
# Assuming script is run from project root, or paths are relative.
# --------------------------------------------------------------------

# Start FastAPI backend (if api directory exists)
if [ -d "api" ]; then
    echo "🐍 Starting FastAPI backend on port 8000 (bind 0.0.0.0)..."
    # Use 0.0.0.0 for cloud/VastAI access, remove --reload in production for stability
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    echo "   Backend PID: $BACKEND_PID"
else
    echo "⚠️  API directory not found - skipping backend startup"
    echo "   Create /api/main.py with FastAPI app"
fi

# Start Next.js frontend (if frontend directory exists)
if [ -d "frontend" ]; then
    echo "🎨 Starting Next.js frontend on port 3000..."

    # Load NVM (if available, for local setup consistency)
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

    cd frontend

    # Check if build exists
    if [ ! -d ".next" ]; then
        echo "   ⚠️  No build found, running npm run build first..."
        npm run build
    fi

    # Use production start mode, not development
    npm run start &
    FRONTEND_PID=$!
    echo "   Frontend PID: $FRONTEND_PID"
    cd ..
else
    echo "⚠️  Frontend directory not found - skipping frontend startup"
    echo "   Create /frontend with Next.js app"
fi

echo ""
echo "=========================================="
echo "✅ Vast.ai/Cloud Services Started!"
echo "=========================================="
echo ""
echo "🌐 Access URLs (use Vast.ai port forwarding or public IP):"
echo "   Frontend: http://0.0.0.0:3000 (access via Vast.ai URL)"
echo "   Backend API: http://0.0.0.0:8000 (access via Vast.ai URL)"
echo "   API Docs: http://0.0.0.0:8000/docs (access via Vast.ai URL)"
echo ""
echo "📊 To monitor services:"
echo "   ps aux | grep -E 'uvicorn|node'"
echo ""
echo "🛑 To stop services:"
echo "   lsof -ti:8000 | xargs kill -9"
echo "   lsof -ti:3000 | xargs kill -9"
echo ""

# Keep script running (important for Docker containers)
wait