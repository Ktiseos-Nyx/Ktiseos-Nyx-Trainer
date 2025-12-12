#!/bin/bash
# VastAI Provisioning Script for Ktiseos-Nyx-Trainer
# This script runs automatically when a VastAI instance starts
# Set via: PROVISIONING_SCRIPT=https://raw.githubusercontent.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/main/vastai_setup.sh

# Note: Removed 'set -e' for more resilient provisioning
# Critical errors are handled explicitly below

provisioning_start() {
    echo "=========================================="
    echo "🚀 Ktiseos-Nyx-Trainer Setup Starting..."
    echo "=========================================="

    # Ensure proper environment setup for VastAI
    export PATH="$PATH:/usr/local/bin:/usr/bin:/bin"

    # Activate virtual environment (VastAI PyTorch base image has this)
    if [ -f "/venv/main/bin/activate" ]; then
        source /venv/main/bin/activate
        echo "✅ Virtual environment activated"
    fi

    # Check for python and ensure it's available
    if ! command -v python &> /dev/null; then
        if command -v python3 &> /dev/null; then
            echo "Creating python alias for python3..."
            alias python=python3
            export PATH="/usr/bin:$PATH"
        else
            echo "❌ Neither python nor python3 found!"
            exit 1
        fi
    fi

    # Handle git safe directory issue when running as root
    export GIT_CONFIG_GLOBAL=/tmp/temporary-git-config
    git config --file $GIT_CONFIG_GLOBAL --add safe.directory '*'

    # Navigate to workspace
    # Note: When using Docker image, code is already copied into /opt/workspace-internal/Ktiseos-Nyx-Trainer
    # If not using Docker (bare provisioning), clone from GitHub
    if [ -d "/opt/workspace-internal/Ktiseos-Nyx-Trainer" ]; then
        echo "📂 Using pre-installed code from Docker image..."
        cd /opt/workspace-internal/Ktiseos-Nyx-Trainer
    elif [ -d "/workspace/Ktiseos-Nyx-Trainer" ]; then
        echo "📂 Repository exists in /workspace, pulling latest changes..."
        cd /workspace/Ktiseos-Nyx-Trainer
        git config --file $GIT_CONFIG_GLOBAL --add safe.directory "$(pwd)"
        git pull || echo "⚠️ Git pull failed, continuing with existing code"
    else
        echo "📥 Cloning repository (fallback for bare provisioning)..."
        cd /workspace
        git clone https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
        cd Ktiseos-Nyx-Trainer
    fi

    # Ensure we're using the right Python
    PYTHON_CMD=$(which python || which python3)
    echo "🐍 Using Python: $PYTHON_CMD"

    # Install Node.js 20+ (needed for Next.js frontend and to resolve dependency compatibility issues)
    echo "📦 Checking Node.js installation..."

    NODEJS_INSTALLED=false
    if command -v node &> /dev/null; then
        CURRENT_VERSION=$(node --version | sed 's/v//' | cut -d. -f1)
        if [ "$CURRENT_VERSION" -ge 20 ]; then
            echo "✅ Node.js $CURRENT_VERSION already installed: $(node --version)"
            NODEJS_INSTALLED=true
        else
            echo "⚠️  Current Node.js version ($CURRENT_VERSION) is too old, need v20+"
        fi
    fi

    # Only install if not already present
    if [ "$NODEJS_INSTALLED" = false ]; then
        echo "📦 Installing Node.js 20+..."
        if command -v apt-get &> /dev/null; then
            # Try to install Node.js 20 using nodesource
            if curl -fsSL https://deb.nodesource.com/setup_20.x | bash - ; then
                # Try installing - but don't fail if there are conflicts
                apt-get install -y nodejs npm || {
                    echo "⚠️  apt-get install failed, trying nvm fallback..."
                    NODEJS_INSTALLED=false
                }
            fi
        fi

        # Fallback to nvm if apt-get failed or not available
        if [ "$NODEJS_INSTALLED" = false ] && ! command -v node &> /dev/null; then
            echo "📦 Trying nvm installation..."
            export NVM_DIR="$HOME/.nvm"
            if [ -s "$NVM_DIR/nvm.sh" ]; then
                \. "$NVM_DIR/nvm.sh"
            else
                # Install nvm first, then install node 20
                curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
                export NVM_DIR="$HOME/.nvm"
                [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
            fi
            nvm install 20 && nvm use 20
        fi
    fi

    # Verify Node.js is available
    if command -v node &> /dev/null; then
        echo "✅ Node.js ready: $(node --version)"
        echo "✅ npm version: $(npm --version)"
    else
        echo "❌ Node.js installation failed - frontend build will be skipped"
    fi

    # Run unified installer (handles all backend dependencies and setup)
    echo "🔧 Running unified installer..."
    if [ -f "installer.py" ]; then
        $PYTHON_CMD installer.py
    else
        echo "⚠️  installer.py not found - falling back to manual dependency installation"

        # Fallback: Install dependencies manually
        echo "🐍 Installing all dependencies..."
        $PYTHON_CMD -m pip install --upgrade pip -v
        if [ -f "requirements.txt" ]; then
            # First try to resolve any version conflicts
            $PYTHON_CMD -m pip install --upgrade setuptools wheel -v
            # Install with conflict resolution (verbose for debugging)
            $PYTHON_CMD -m pip install -r requirements.txt --no-cache-dir -v
        fi
    fi

    # Setup Next.js Frontend
    if [ -d "frontend" ]; then
        if command -v node &> /dev/null && command -v npm &> /dev/null; then
            echo "🎨 Setting up Next.js frontend..."
            cd frontend

            # Ensure Node.js is available
            export NVM_DIR="$HOME/.nvm"
            [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

            # Check node version and install dependencies
            echo "Node version: $(node --version)"
            echo "NPM version: $(npm --version)"

            echo "   Installing npm packages..."
            npm ci --prefer-offline || npm install || {
                echo "⚠️  npm install failed, continuing anyway..."
            }

            # Build for production
            echo "🏗️  Building Next.js app..."
            npm run build || {
                echo "⚠️  Frontend build failed, services may not work correctly"
            }

            cd ..
        else
            echo "⚠️  Node.js/npm not available - skipping frontend setup"
        fi
    else
        echo "⚠️  Frontend directory not found - skipping Next.js setup"
    fi

    # Make startup scripts executable
    if [ -f "start_services_vastai.sh" ]; then
        chmod +x start_services_vastai.sh
    fi

    if [ -f "start_services_local.sh" ]; then
        chmod +x start_services_local.sh
    fi

    # Configure git for root usage
    git config --global --add safe.directory $(pwd)

    # Start services directly (simpler, won't interfere with VastAI Portal)
    echo ""
    echo "🚀 Starting services..."

    # Activate virtual environment
    source /venv/main/bin/activate 2>/dev/null || true

    # Clean up any existing processes on our ports
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true

    # Create log directory
    mkdir -p /workspace/logs

    # Start backend
    echo "   Starting FastAPI backend on port 8000..."
    nohup python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 > /workspace/logs/backend.log 2>&1 &
    echo $! > /workspace/logs/backend.pid

    # Start frontend
    echo "   Starting Next.js frontend on port 3000..."
    cd frontend
    nohup npm run start > /workspace/logs/frontend.log 2>&1 &
    echo $! > /workspace/logs/frontend.pid
    cd ..

    # Give services a moment to start
    sleep 3

    # Check if services are running
    if ps -p $(cat /workspace/logs/backend.pid 2>/dev/null) > /dev/null 2>&1; then
        echo "   ✅ Backend started (PID: $(cat /workspace/logs/backend.pid))"
    else
        echo "   ⚠️  Backend may have failed to start - check /workspace/logs/backend.log"
    fi

    if ps -p $(cat /workspace/logs/frontend.pid 2>/dev/null) > /dev/null 2>&1; then
        echo "   ✅ Frontend started (PID: $(cat /workspace/logs/frontend.pid))"
    else
        echo "   ⚠️  Frontend may have failed to start - check /workspace/logs/frontend.log"
    fi

    echo ""
    echo "=========================================="
    echo "✅ Setup Complete!"
    echo "=========================================="
    echo ""
    echo "🌐 Access your applications via VastAI portal links:"
    echo "   - Frontend: Next.js UI (port 3000)"
    echo "   - Backend: FastAPI (port 8000)"
    echo "   - Jupyter: File management (port 8080)"
    echo "   - TensorBoard: Training monitoring (port 6006)"
    echo ""
    echo "📋 Service logs:"
    echo "   - Backend:  /workspace/logs/backend.log"
    echo "   - Frontend: /workspace/logs/frontend.log"
    echo ""
    echo "🔧 To restart services:"
    echo "   - Backend:  kill \$(cat /workspace/logs/backend.pid) && cd /workspace/Ktiseos-Nyx-Trainer && nohup python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 > /workspace/logs/backend.log 2>&1 &"
    echo "   - Frontend: kill \$(cat /workspace/logs/frontend.pid) && cd /workspace/Ktiseos-Nyx-Trainer/frontend && nohup npm run start > /workspace/logs/frontend.log 2>&1 &"
    echo ""
}

# Check if provisioning should be skipped
if [[ ! -f /.noprovisioning ]]; then
    provisioning_start
fi
