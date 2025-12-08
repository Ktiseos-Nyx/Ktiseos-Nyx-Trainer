#!/bin/bash
# Service Startup Script for Ktiseos-Nyx-Trainer
# --------------------------------------------------------------------
# FIX: "Homing Pigeon" Logic
# This ensures the script works no matter where you call it from.
# --------------------------------------------------------------------

# Get the directory where THIS script is actually located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Move to that directory (Assuming this script is in the ROOT of your project)
cd "$SCRIPT_DIR"

# (Optional: If you move this script into a /scripts folder later, use this line instead:)
# cd "$SCRIPT_DIR/.."

echo "=========================================="
echo "🚀 Starting Ktiseos-Nyx-Trainer (Production Mode)..."
echo "📂 Working Directory: $(pwd)"
echo "=========================================="

set -e

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
# Step 1: Python / Backend
# --------------------------------------------------------------------
if [ -d "api" ]; then
    echo "🐍 Starting FastAPI backend..."
    # Note: Removed --reload because you said 'Production Build'.
    # Reload is for dev. If you want dev, add --reload back.
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    echo "   Backend PID: $BACKEND_PID"
else
    echo "❌ CRITICAL: 'api' directory not found in $(pwd)"
    exit 1
fi

# --------------------------------------------------------------------
# Step 2: Next.js / Frontend
# --------------------------------------------------------------------
if [ -d "frontend" ]; then
    echo "🎨 Preparing Frontend..."

    # Load NVM if it exists (Standard safety check)
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

    cd frontend

    # CRITICAL CHECK FOR PRODUCTION
    # npm start fails if you haven't built it first.
    if [ ! -d ".next" ]; then
        echo "⚠️  No build found! Running 'npm run build' first..."
        echo "    (This might take a minute, grab a drink)"
        npm run build
    fi

    echo "🚀 Starting Next.js Production Server..."
    npm start -- -p 3000 &
    FRONTEND_PID=$!
    echo "   Frontend PID: $FRONTEND_PID"

    # Go back to root just to be safe
    cd ..
else
    echo "❌ CRITICAL: 'frontend' directory not found in $(pwd)"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Services Running!"
echo "=========================================="
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo ""
echo "🛑 Press CTRL+C to stop both services"

# This magic wait command allows you to kill the script with Ctrl+C
# and it will shut down the background processes cleanly.
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
