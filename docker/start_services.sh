#!/bin/bash
# Quick service starter for VastAI Jupyter+SSH mode
# Run this after SSH login: bash /workspace/Ktiseos-Nyx-Trainer/docker/start_services.sh

echo "=================================================="
echo "  Ktiseos Nyx - Service Starter"
echo "=================================================="
echo ""

# Check if we're in the right place
if [ ! -d "/opt/workspace-internal/Ktiseos-Nyx-Trainer" ]; then
    echo "❌ Repository not found at /opt/workspace-internal/Ktiseos-Nyx-Trainer"
    echo ""
    echo "Running first-time setup..."
    /startup.sh
    exit 0
fi

cd /opt/workspace-internal/Ktiseos-Nyx-Trainer

# Create logs directory
mkdir -p /opt/workspace-internal/logs

# Check if services are already running
BACKEND_RUNNING=$(pgrep -f "uvicorn api.main:app" || echo "")
FRONTEND_RUNNING=$(pgrep -f "npm run start" || echo "")

if [ -n "$BACKEND_RUNNING" ]; then
    echo "⚠️  Backend already running (PID: $BACKEND_RUNNING)"
    echo "   To restart, run: pkill -f uvicorn && bash $0"
else
    echo "🚀 Starting Backend API..."
    nohup uvicorn api.main:app --host 0.0.0.0 --port 8000 > /opt/workspace-internal/logs/backend.log 2>&1 &
    BACKEND_PID=$!
    echo "   ✅ Backend started (PID: $BACKEND_PID)"
fi

if [ -n "$FRONTEND_RUNNING" ]; then
    echo "⚠️  Frontend already running (PID: $FRONTEND_RUNNING)"
    echo "   To restart, run: pkill -f 'npm run start' && bash $0"
else
    echo "🚀 Starting Frontend..."
    cd frontend

    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        echo "   📦 Installing dependencies (first run)..."
        npm install
    fi

    # Check if .next exists
    if [ ! -d ".next" ]; then
        echo "   🔨 Building Next.js app (first run)..."
        npm run build
    fi

    nohup npm run start > /opt/workspace-internal/logs/frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo "   ✅ Frontend started (PID: $FRONTEND_PID)"
    cd ..
fi

echo ""
echo "=================================================="
echo "  Services Ready!"
echo "=================================================="
echo ""
echo "📡 Access your services:"
echo "   Frontend:  http://$(hostname -I | awk '{print $1}'):3000"
echo "   Backend:   http://$(hostname -I | awk '{print $1}'):8000"
echo "   API Docs:  http://$(hostname -I | awk '{print $1}'):8000/docs"
echo ""
echo "📝 View logs:"
echo "   Backend:   tail -f /opt/workspace-internal/logs/backend.log"
echo "   Frontend:  tail -f /opt/workspace-internal/logs/frontend.log"
echo ""
echo "🛑 Stop services:"
echo "   pkill -f uvicorn"
echo "   pkill -f 'npm run start'"
echo ""
echo "=================================================="
