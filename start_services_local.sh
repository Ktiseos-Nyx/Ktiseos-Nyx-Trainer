#!/bin/bash
# Service Startup Script for Ktiseos-Nyx-Trainer (Local Development)
# This script starts both the FastAPI backend and Next.js frontend, binding to 127.0.0.1

echo "=========================================="
echo "🚀 Starting Ktiseos-Nyx-Trainer Services (Local)..."
echo "=========================================="

# --------------------------------------------------------------------
# Step 0: Clean up any existing processes using the ports
# --------------------------------------------------------------------
echo "🧹 Cleaning up any existing processes on ports 8000 and 3000..."
lsof -ti:8000 2>/dev/null | xargs kill -9 2>/dev/null || true
lsof -ti:3000 2>/dev/null | xargs kill -9 2>/dev/null || true

# Also kill any existing uvicorn or next processes to prevent conflicts
pkill -f "uvicorn.*api.main" 2>/dev/null || true
pkill -f "node.*next" 2>/dev/null || true

sleep 3  # Give time for processes to terminate

# --------------------------------------------------------------------
# Step 1: Run local installer if available (for environment verification)
# --------------------------------------------------------------------
if [ -f "installer_local_linux.py" ]; then
    echo "⚙️ Running local Linux installer (verification only)..."
    if ! python3 installer_local_linux.py --skip-install; then
        echo "❌ Installer verification failed! Please run installer manually first."
        exit 1
    fi
else
    echo "ℹ️ No local installer found — assuming environment is already set up."
    echo "   To install dependencies, run:"
    echo "      Linux/WSL: python installer_local_linux.py"
    echo ""
fi

# --------------------------------------------------------------------
# Step 2: Start Services
# --------------------------------------------------------------------

# Start FastAPI backend (if api directory exists)
if [ -d "api" ]; then
    echo "🐍 Starting FastAPI backend on port 8000 (bind 127.0.0.1)..."
    python3 -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload &
    BACKEND_PID=$!
    echo "   Backend PID: $BACKEND_PID"
else
    echo "⚠️  API directory not found - skipping backend startup"
    echo "   Create /api/main.py with FastAPI app"
fi

# Start Next.js frontend (if frontend directory exists)
if [ -d "frontend" ]; then
    echo "🎨 Preparing Next.js frontend..."

    # Load NVM (if available)
    export NVM_DIR="$HOME/.nvm"
    # shellcheck disable=SC1091
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

    # shellcheck disable=SC2164
    cd frontend

    # Check if production build exists
    if [ ! -d ".next" ]; then
        echo "📦 No production build found - building Next.js app..."
        npm run build
        if [ $? -ne 0 ]; then
            echo "❌ Build failed! Please fix errors and try again."
            cd ..
            exit 1
        fi
        echo "✅ Build complete!"
    else
        echo "📦 Using existing production build (run 'npm run build' to rebuild)"
    fi

    # Start production server
    echo "🚀 Starting Next.js production server on port 3000..."
    NODE_ENV=production PORT=3000 npm run start &
    FRONTEND_PID=$!
    echo "   Frontend PID: $FRONTEND_PID"
    # shellcheck disable=SC2103
    cd ..
else
    echo "⚠️  Frontend directory not found - skipping frontend startup"
    echo "   Create /frontend with Next.js app"
fi

echo ""
echo "=========================================="
echo "✅ Local Services Started!"
echo "=========================================="
echo ""
echo "🌐 Access URLs:"
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

# Keep script running (important for foreground process)
wait
