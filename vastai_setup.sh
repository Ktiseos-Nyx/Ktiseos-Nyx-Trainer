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

    # Check Node.js version (Next.js requires 18.18+)
    echo "📦 Checking Node.js installation..."

    if ! command -v node &> /dev/null; then
        echo "❌ ERROR: Node.js not found!"
        echo "   VastAI PyTorch images should have Node.js pre-installed."
        echo "   Please use a VastAI image with Node.js 18+ included."
        exit 1
    fi

    CURRENT_VERSION=$(node --version | sed 's/v//' | cut -d. -f1)
    if [ "$CURRENT_VERSION" -lt 18 ]; then
        echo "❌ ERROR: Node.js $CURRENT_VERSION is too old (need 18+)"
        echo "   Found at: $(which node)"
        echo "   Please use a VastAI image with Node.js 18+ included."
        exit 1
    fi

    echo "✅ Node.js $CURRENT_VERSION ready: $(node --version) at $(which node)"
    echo "✅ npm version: $(npm --version)"

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

# Clean up any existing processes on our ports (only if they're python/node processes)
# This prevents accidentally killing VastAI infrastructure
echo "[$(date)] Checking for existing services on ports 8000 and 3000..." | tee -a /workspace/logs/supervisor.log
pkill -f "uvicorn api.main:app" 2>/dev/null || true
pkill -f "next-server.*3000" 2>/dev/null || true
sleep 1

# Start backend
echo "[$(date)] Starting FastAPI backend on port 8000..." | tee -a /workspace/logs/supervisor.log
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 2>&1 | tee -a /workspace/logs/backend.log &
BACKEND_PID=$!

# Give backend a moment to start
sleep 2

# Start frontend
echo "[$(date)] Starting Next.js frontend on port 3000..." | tee -a /workspace/logs/supervisor.log
cd frontend || exit 1
npm run start 2>&1 | tee -a /workspace/logs/frontend.log &
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
    echo "   - Jupyter: File management (port 8080)"
    echo "   - TensorBoard: Training monitoring (port 6006)"
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
