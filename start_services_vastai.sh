#!/bin/bash
# Service Startup Script for Ktiseos-Nyx-Trainer (Vast.ai / Cloud Deployment)
# This script starts both the FastAPI backend and Next.js frontend, binding to 0.0.0.0

set -e

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

# Start Next.js frontend (if frontend directory exists)
if [ -d "frontend" ]; then
    # Try to find Node.js in common locations if not in PATH
    if ! command -v node &> /dev/null; then
        echo "   Node.js not in PATH, searching common locations..."
        for node_path in /opt/nvm/versions/node/*/bin /usr/bin /usr/local/bin ~/.nvm/versions/node/*/bin; do
            if [ -f "$node_path/node" ]; then
                echo "   Found Node.js at: $node_path/node"
                export PATH="$node_path:$PATH"
                break
            fi
        done
    fi

    # Check if Node.js and npm are available
    if command -v node &> /dev/null && command -v npm &> /dev/null; then
        FRONTEND_PORT="${FRONTEND_PORT:-13000}"
        echo "🎨 Starting Next.js frontend on port $FRONTEND_PORT..."

        # Load NVM (if available, for local setup consistency)
        export NVM_DIR="$HOME/.nvm"
        # shellcheck disable=SC1091
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

        cd frontend

        # Check if build exists
        if [ ! -d ".next" ]; then
            echo "   ⚠️  No build found, running npm run build first..."
            # Ensure node_modules exists (may need install if lockfile was platform-specific)
            if [ ! -d "node_modules" ] || [ ! -d "node_modules/next" ]; then
                echo "   📦 node_modules missing or incomplete, installing..."
                rm -f package-lock.json
                npm install --legacy-peer-deps || npm install --legacy-peer-deps --force || true
            fi
            npm run build || {
                echo "❌ Frontend build failed - skipping frontend startup"
                cd ..
                echo "⚠️  Frontend unavailable. Backend API will still work on port $BACKEND_PORT."
                echo ""
            }
        fi

        # Use production start mode, not development (only if build succeeded)
        if [ -d ".next" ]; then
            PORT=$FRONTEND_PORT BACKEND_PORT=$BACKEND_PORT NODE_ENV=production npm run start &
            FRONTEND_PID=$!
            echo "   Frontend PID: $FRONTEND_PID"
            cd ..
        fi
    else
        echo "⚠️  Node.js/npm not found - skipping frontend startup"
        echo "   Frontend unavailable. Backend API will still work on port ${BACKEND_PORT:-18000}."
    fi
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
