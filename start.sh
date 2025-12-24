#!/bin/bash
# Service Startup Script for Ktiseos-Nyx-Trainer (Production-Style Local)
# --------------------------------------------------------------------
# "Homing Pigeon" Logic + UTF-8 Safety for Git Bash / WSL
# --------------------------------------------------------------------

# Ensure UTF-8 for emoji and console compatibility (critical for Git Bash on Windows)
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "🚀 Starting Ktiseos-Nyx-Trainer (Production Mode)..."
echo "📂 Working Directory: $(pwd)"
echo "=========================================="

# --------------------------------------------------------------------
# Dependency Check
# --------------------------------------------------------------------
echo "🔍 Verifying dependencies..."

if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Install Python 3.10+ and try again."
    exit 1
fi

if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Install Node.js 18+ and try again."
    exit 1
fi

if ! python -c "import uvicorn, fastapi" 2>/dev/null; then
    echo "❌ Python dependencies missing."
    echo "   → Run installer first (e.g., installer_local_linux.py)"
    exit 1
fi

# --------------------------------------------------------------------
# Cleanup
# --------------------------------------------------------------------
echo "🧹 Cleaning up ports 8000 and 3000..."
lsof -ti:8000 2>/dev/null | xargs kill -9 2>/dev/null || true
lsof -ti:3000 2>/dev/null | xargs kill -9 2>/dev/null || true
pkill -f "uvicorn.*api.main" 2>/dev/null || true
pkill -f "node.*next" 2>/dev/null || true
sleep 3

# --------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------
if [ -d "api" ]; then
    echo "🐍 Starting FastAPI backend on 0.0.0.0:8000..."
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    echo "   Backend PID: $BACKEND_PID"
else
    echo "❌ CRITICAL: 'api' directory missing"
    exit 1
fi

# --------------------------------------------------------------------
# Frontend
# --------------------------------------------------------------------
if [ -d "frontend" ]; then
    echo "🎨 Starting Next.js frontend..."

    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

    cd frontend

    if [ ! -d ".next" ]; then
        echo "⚠️  Building frontend (first run)..."
        npm run build
    fi

    echo "🚀 Starting Next.js production server..."
    PORT=3000 npm start &
    FRONTEND_PID=$!
    echo "   Frontend PID: $FRONTEND_PID"
    cd ..
else
    echo "❌ CRITICAL: 'frontend' directory missing"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Production Services Running!"
echo "=========================================="
echo "   Backend:  http://0.0.0.0:8000"
echo "   Frontend: http://0.0.0.0:3000"
echo ""
echo "🛑 Press CTRL+C to stop"

# Graceful shutdown on Ctrl+C
trap "echo -e '\n🛑 Shutting down...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
