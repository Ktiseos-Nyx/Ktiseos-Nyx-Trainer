#!/bin/bash
# Service Startup Script for Ktiseos-Nyx-Trainer
# This script starts both the FastAPI backend and Next.js frontend

set -e

echo "=========================================="
echo "🚀 Starting Ktiseos-Nyx-Trainer Services..."
echo "=========================================="

# Note: VastAI PyTorch template already has venv activated
# No need to activate it manually!

# Navigate to project directory
cd /workspace/Ktiseos-Nyx-Trainer

# Start FastAPI backend (if api directory exists)
if [ -d "api" ]; then
    echo "🐍 Starting FastAPI backend on port 8000..."
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
    BACKEND_PID=$!
    echo "   Backend PID: $BACKEND_PID"
else
    echo "⚠️  API directory not found - skipping backend startup"
    echo "   Create /api/main.py with FastAPI app"
fi

# Start Next.js frontend (if frontend directory exists)
if [ -d "frontend" ]; then
    echo "🎨 Starting Next.js frontend on port 3000..."

    # Load NVM
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

    cd frontend
    npm start -- -p 3000 &
    FRONTEND_PID=$!
    echo "   Frontend PID: $FRONTEND_PID"
    cd ..
else
    echo "⚠️  Frontend directory not found - skipping frontend startup"
    echo "   Create /frontend with Next.js app"
fi

echo ""
echo "=========================================="
echo "✅ Services Started!"
echo "=========================================="
echo ""
echo "🌐 Access URLs (use VastAI port forwarding):"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "📊 To monitor services:"
echo "   ps aux | grep -E 'uvicorn|node'"
echo ""
echo "🛑 To stop services:"
echo "   pkill -f uvicorn"
echo "   pkill -f 'node.*next'"
echo ""

# Keep script running (important for Docker containers)
wait
