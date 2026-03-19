#!/bin/bash
# Service Startup Script for Ktiseos-Nyx-Trainer (Vast.ai / Cloud Deployment)
# This script starts both the FastAPI backend and Next.js frontend, binding to 0.0.0.0

# Note: no set -e — frontend build failures must not abort backend startup

echo "=========================================="
echo "🚀 Starting Ktiseos-Nyx-Trainer Services (Vast.ai/Cloud)..."
echo "=========================================="

# --------------------------------------------------------------------
# Step 1: Verify environment
# Note: VastAI provisioning (vastai_setup.sh) handles full install.
# By the time this script runs, packages should already be installed.
# --------------------------------------------------------------------
echo "⚙️ Verifying Python environment..."
if [ -f "installer.py" ]; then
    python installer.py --skip-install 2>/dev/null || echo "   (installer check skipped)"
else
    echo "   ℹ️  No installer found - assuming environment is pre-configured"
fi

# --------------------------------------------------------------------
# Step 2: Start Services
# Assuming script is run from project root, or paths are relative.
# --------------------------------------------------------------------

# Start FastAPI backend (if api directory exists)
if [ -d "api" ]; then
    # VastAI's Caddy binds ports 3000/8000 and reverse-proxies to 13000/18000.
    # Our services must listen on the proxy-target ports, not the Caddy-owned ports.
    BACKEND_PORT="${BACKEND_PORT:-18000}"
    echo "🐍 Starting FastAPI backend on port $BACKEND_PORT (bind 0.0.0.0)..."
    python -m uvicorn api.main:app --host 0.0.0.0 --port "$BACKEND_PORT" &
    BACKEND_PID=$!
    echo "   Backend PID: $BACKEND_PID"
else
    echo "⚠️  API directory not found - skipping backend startup"
    echo "   Create /api/main.py with FastAPI app"
fi

# Start Next.js frontend
if [ -d "frontend" ]; then
    FRONTEND_PORT="${FRONTEND_PORT:-13000}"

    # If .next/ is missing, run the Python frontend installer to build it.
    # install_frontend.py handles Node.js auto-upgrade, npm install, and build.
    if [ ! -d "frontend/.next" ]; then
        echo "🎨 No frontend build found — running install_frontend.py..."
        if [ -f "install_frontend.py" ]; then
            python install_frontend.py || echo "⚠️  Frontend setup failed — backend-only mode."
        else
            echo "⚠️  install_frontend.py not found — frontend unavailable."
        fi
    fi

    # Start if build now exists
    if [ -d "frontend/.next" ] && command -v node &> /dev/null; then
        # Validate Node version meets >=20.19.0 requirement before launching server.js
        NODE_VER=$(node --version 2>/dev/null | sed 's/^v//')
        NODE_MAJOR=$(echo "$NODE_VER" | cut -d. -f1)
        NODE_MINOR=$(echo "$NODE_VER" | cut -d. -f2)
        if [ -z "$NODE_MAJOR" ] || [ "$NODE_MAJOR" -lt 20 ] || \
           { [ "$NODE_MAJOR" -eq 20 ] && [ "${NODE_MINOR:-0}" -lt 19 ]; }; then
            echo "⚠️  Node.js ${NODE_VER} does not meet >=20.19.0 requirement — aborting frontend startup."
            echo "   Run: python install_frontend.py --force"
        else
            echo "🎨 Starting Next.js frontend on port $FRONTEND_PORT..."
            cd frontend
            PORT=$FRONTEND_PORT BACKEND_PORT=$BACKEND_PORT NODE_ENV=production node server.js &
            FRONTEND_PID=$!
            echo "   Frontend PID: $FRONTEND_PID"
            cd ..
        fi
    else
        echo "⚠️  Frontend unavailable. Backend API will still work on port ${BACKEND_PORT:-18000}."
        echo "   To build: python install_frontend.py"
    fi
else
    echo "⚠️  Frontend directory not found - skipping frontend startup"
fi

echo ""
echo "=========================================="
echo "✅ Vast.ai/Cloud Services Started!"
echo "=========================================="
echo ""
echo "🌐 Access URLs (use Vast.ai port forwarding or public IP):"
echo "   Frontend: http://0.0.0.0:${FRONTEND_PORT:-13000} (Caddy proxies from :3000)"
echo "   Backend API: http://0.0.0.0:${BACKEND_PORT:-18000} (Caddy proxies from :8000)"
echo "   API Docs: http://0.0.0.0:${BACKEND_PORT:-18000}/docs"
echo ""
echo "📊 To monitor services:"
echo "   ps aux | grep -E 'uvicorn|node'"
echo ""
echo "🛑 To stop services:"
echo "   lsof -ti:${BACKEND_PORT:-18000} | xargs kill -9"
echo "   lsof -ti:${FRONTEND_PORT:-13000} | xargs kill -9"
echo ""

# Keep script running (important for Docker containers)
wait
