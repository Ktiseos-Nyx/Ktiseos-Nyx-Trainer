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
        # shellcheck disable=SC1091
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
        # shellcheck disable=SC2164
        cd /opt/workspace-internal/Ktiseos-Nyx-Trainer
    elif [ -d "/workspace/Ktiseos-Nyx-Trainer" ]; then
        echo "📂 Repository exists in /workspace, pulling latest changes..."
        # shellcheck disable=SC2164
        cd /workspace/Ktiseos-Nyx-Trainer
        git config --file $GIT_CONFIG_GLOBAL --add safe.directory "$(pwd)"
        git pull || echo "⚠️ Git pull failed, continuing with existing code"
    else
        echo "📥 Cloning repository (fallback for bare provisioning)..."
        # shellcheck disable=SC2164
        cd /workspace
        git clone https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
        # shellcheck disable=SC2164
        cd Ktiseos-Nyx-Trainer
    fi

    # Ensure we're using the right Python
    PYTHON_CMD=$(which python || which python3)
    echo "🐍 Using Python: $PYTHON_CMD"

    # Check Node.js version (Next.js requires 18.18+)
    echo "📦 Checking Node.js installation..."

    # Try to find Node.js in common locations if not in PATH
    if ! command -v node &> /dev/null; then
        echo "   Node.js not in PATH, searching common locations..."
        NODE_FOUND=false

        for node_path in /opt/nvm/versions/node/*/bin /usr/bin /usr/local/bin ~/.nvm/versions/node/*/bin; do
            if [ -f "$node_path/node" ]; then
                echo "   Found Node.js at: $node_path/node"
                export PATH="$node_path:$PATH"
                NODE_FOUND=true
                break
            fi
        done

        if [ "$NODE_FOUND" = false ]; then
            echo "⚠️  WARNING: Node.js not found!"
            echo "   Frontend will be unavailable (Next.js requires Node.js 18+)"
            echo "   Backend API will still work normally."
            echo ""
            echo "   Continuing with backend-only setup..."
            SKIP_FRONTEND=true
        fi
    fi

    # Validate Node.js version if found
    if [ "$SKIP_FRONTEND" != true ] && command -v node &> /dev/null; then
        CURRENT_VERSION=$(node --version | sed 's/v//' | cut -d. -f1)
        if [ "$CURRENT_VERSION" -lt 18 ]; then
            echo "⚠️  WARNING: Node.js $CURRENT_VERSION is too old (need 18+)"
            echo "   Found at: $(which node)"
            echo "   Frontend will be unavailable."
            echo "   Backend API will still work normally."
            echo ""
            SKIP_FRONTEND=true
        else
            echo "✅ Node.js $CURRENT_VERSION ready: $(node --version) at $(which node)"
            echo "✅ npm version: $(npm --version)"
        fi
    fi

    # Run unified installer (handles all backend dependencies and setup)
    echo "🔧 Running installer..."
    if [ -f "installer.py" ]; then
        $PYTHON_CMD installer.py
    else
        echo "⚠️  installer.py not found - falling back to manual dependency installation"
        # Fallback: Install dependencies manually
        echo "🐍 Installing all dependencies..."
        $PYTHON_CMD -m pip install --upgrade pip -v
        # Use VastAI-specific requirements (no PyTorch - pre-installed in base image)
        if [ -f "requirements_cloud.txt" ]; then
            # First try to resolve any version conflicts
            $PYTHON_CMD -m pip install --upgrade setuptools wheel -v
            # Install with conflict resolution (verbose for debugging)
            $PYTHON_CMD -m pip install -r requirements_cloud.txt --no-cache-dir -v
        elif [ -f "requirements.txt" ]; then
            # Fallback to old requirements.txt if cloud version doesn't exist yet
            echo "⚠️  requirements_cloud.txt not found, using requirements.txt"
            $PYTHON_CMD -m pip install -r requirements.txt --no-cache-dir -v
        fi
    fi

    # Setup Next.js Frontend
    if [ -d "frontend" ] && [ "$SKIP_FRONTEND" != true ]; then
        if command -v node &> /dev/null && command -v npm &> /dev/null; then
            echo "🎨 Setting up Next.js frontend..."
            # shellcheck disable=SC2164
            cd frontend

            # Ensure Node.js is available
            export NVM_DIR="$HOME/.nvm"
            # shellcheck disable=SC1091
            [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

            # Check node version and install dependencies
            echo "Node version: $(node --version)"
            echo "NPM version: $(npm --version)"

            echo "   Installing npm packages..."
            # Remove stale lockfile from different OS to avoid platform-specific dep errors
            # (e.g. lightningcss-win32-x64-msvc fails on Linux)
            if [ -f "package-lock.json" ]; then
                echo "   Removing package-lock.json (regenerating for this platform)..."
                rm -f package-lock.json
            fi
            npm install --legacy-peer-deps || npm install --legacy-peer-deps --force || {
                echo "⚠️  npm install failed, continuing anyway..."
            }

            # Build for production
            echo "🏗️  Building Next.js app..."
            npm run build || {
                echo "⚠️  Frontend build failed, services may not work correctly"
            }

            # shellcheck disable=SC2103
            cd ..
        else
            echo "⚠️  Node.js/npm not available - skipping frontend setup"
        fi
    elif [ "$SKIP_FRONTEND" = true ]; then
        echo "⏭️  Skipping frontend setup (Node.js not available)"
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
    # shellcheck disable=SC2046
    git config --global --add safe.directory $(pwd)

    # Create log directory
    mkdir -p /workspace/logs

    # Create supervisor config for auto-restart
    echo ""
    echo "📝 Creating supervisor config for auto-restart..."

    mkdir -p /opt/supervisor-scripts

    cat > /opt/supervisor-scripts/ktiseos-nyx.sh << 'EOL'
#!/bin/bash
# Supervisor startup script for Ktiseos-Nyx services

# Activate virtual environment
source /venv/main/bin/activate 2>/dev/null || true

# Ensure node is in PATH (check common locations if not found)
if ! command -v node &> /dev/null; then
    # Check common node installation paths
    for node_path in /opt/nvm/versions/node/*/bin /usr/bin /usr/local/bin; do
        if [ -f "$node_path/node" ]; then
            export PATH="$node_path:$PATH"
            break
        fi
    done
fi

# Navigate to project directory
if [ ! -d /workspace/Ktiseos-Nyx-Trainer ]; then
    echo "[$(date)] ERROR: Project directory not found at /workspace/Ktiseos-Nyx-Trainer" | tee -a /workspace/logs/supervisor.log
    exit 1
fi
cd /workspace/Ktiseos-Nyx-Trainer

# VastAI's Caddy binds ports 3000/8000 and reverse-proxies to 13000/18000.
# Our services must listen on the proxy-target ports, not the Caddy-owned ports.
BACKEND_PORT="${BACKEND_PORT:-18000}"
FRONTEND_PORT="${FRONTEND_PORT:-13000}"

# Clean up any existing processes on our ports (only if they're python/node processes)
# This prevents accidentally killing VastAI infrastructure (Caddy on 3000/8000)
echo "[$(date)] Checking for existing services on ports $BACKEND_PORT and $FRONTEND_PORT..." | tee -a /workspace/logs/supervisor.log
pkill -f "uvicorn api.main:app" 2>/dev/null || true
pkill -f "next-server" 2>/dev/null || true
sleep 1

# Start backend
echo "[$(date)] Starting FastAPI backend on port $BACKEND_PORT..." | tee -a /workspace/logs/supervisor.log
python -m uvicorn api.main:app --host 0.0.0.0 --port "$BACKEND_PORT" 2>&1 | tee -a /workspace/logs/backend.log &
BACKEND_PID=$!

# Give backend a moment to start
sleep 2

# Start frontend (using custom server with WebSocket proxy)
echo "[$(date)] Starting Next.js frontend on port $FRONTEND_PORT..." | tee -a /workspace/logs/supervisor.log
cd frontend || exit 1
PORT=$FRONTEND_PORT BACKEND_PORT=$BACKEND_PORT NODE_ENV=production node server.js 2>&1 | tee -a /workspace/logs/frontend.log &
FRONTEND_PID=$!

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
EOL

    chmod +x /opt/supervisor-scripts/ktiseos-nyx.sh

    cat > /etc/supervisor/conf.d/ktiseos-nyx.conf << 'EOL'
[program:ktiseos-nyx]
command=/opt/supervisor-scripts/ktiseos-nyx.sh
directory=/workspace/Ktiseos-Nyx-Trainer
autostart=true
autorestart=true
startsecs=10
stopasgroup=true
killasgroup=true
stdout_logfile=/workspace/logs/supervisor.log
redirect_stderr=true
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=3
EOL

    echo "   ✅ Supervisor config created"

    echo ""
    echo "=========================================="
    echo "✅ Setup Complete!"
    echo "=========================================="
    echo ""
    echo "🚀 Activating supervisor to start services..."

    # Activate supervisor NOW (at the very end, after everything is ready)
    if command -v supervisorctl &> /dev/null; then
        supervisorctl reread
        supervisorctl update

        # Give services a moment to start
        sleep 5

        # Check if services started
        if supervisorctl status ktiseos-nyx | grep -q RUNNING; then
            echo "   ✅ Services started successfully!"
        else
            echo "   ⚠️  Services may not have started - check logs"
        fi
    else
        echo "   ⚠️  supervisorctl not found - services won't auto-start"
    fi

    echo ""
    echo "🌐 Access your applications via VastAI portal links:"
    echo "   - Frontend: Next.js UI (port 3000)"
    echo "   - Backend: FastAPI (port 8000)"
    echo "   - TensorBoard: Training monitoring (port 6006, auto-configured by VastAI)"
    echo ""
    echo "♻️  Auto-restart is enabled!"
    echo "   - Supervisor will automatically restart services if they crash"
    echo ""
    echo "📋 Service logs:"
    echo "   - Backend:    /workspace/logs/backend.log"
    echo "   - Frontend:   /workspace/logs/frontend.log"
    echo "   - Supervisor: /workspace/logs/supervisor.log"
    echo ""
    echo "🔧 Manual service control:"
    echo "   - Restart:  supervisorctl restart ktiseos-nyx"
    echo "   - Status:   supervisorctl status"
    echo "   - Stop:     supervisorctl stop ktiseos-nyx"
    echo "   - Start:    supervisorctl start ktiseos-nyx"
    echo ""
}

# Check if provisioning should be skipped
if [[ ! -f /.noprovisioning ]]; then
    provisioning_start
fi
