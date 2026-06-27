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

    # Reinstall torchaudio from the CUDA-matching PyTorch index.
    # The Vast base image pre-installs torchaudio from the cu130 index (or later),
    # which won't load on a CUDA 12.x container (libcudart.so version mismatch).
    # Re-install torchaudio as a cu126 wheel. The host MACHINE's newer CUDA (vs our 12.6.3 template)
    # leaves torchaudio as a cu13 build (wants libcudart.so.13) that won't load against cu126 torch ->
    # ComfyUI dies on `import torchaudio`. (It's the host machine, NOT the base image.) We standardize
    # on CUDA 12.6, so point straight at the cu126 index (a cu126 wheel works on any 12.x box). --no-deps
    # because torchaudio's wheel hard-pins `torch==` (would otherwise downgrade torch). Mirrored inline in
    # fetch-restart.sh (this copy runs pre-clone; the restart path re-runs the same line) -- keep in sync.
    pip install --force-reinstall --no-deps torchaudio --index-url https://download.pytorch.org/whl/cu126 \
        || echo "[setup] cu126 torchaudio reinstall failed (non-fatal)"

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
        PULL_OUTPUT=$(git pull 2>&1)
        PULL_EXIT=$?
        echo "$PULL_OUTPUT"
        if [ $PULL_EXIT -ne 0 ]; then
            echo "⚠️ Git pull failed, continuing with existing code"
        elif ! echo "$PULL_OUTPUT" | grep -q "Already up to date"; then
            PULLED=true
        fi
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

    # Run unified installer (handles all backend dependencies and setup)
    echo "🔧 Running backend installer..."
    if [ -f "installer.py" ]; then
        $PYTHON_CMD installer.py
    else
        echo "⚠️  installer.py not found - falling back to manual dependency installation"
        echo "🐍 Installing all dependencies..."
        $PYTHON_CMD -m pip install --upgrade pip -v
        if [ -f "requirements_cloud.txt" ]; then
            $PYTHON_CMD -m pip install --upgrade setuptools wheel -v
            $PYTHON_CMD -m pip install -r requirements_cloud.txt --no-cache-dir -v
        elif [ -f "requirements.txt" ]; then
            echo "⚠️  requirements_cloud.txt not found, using requirements.txt"
            $PYTHON_CMD -m pip install -r requirements.txt --no-cache-dir -v
        fi
    fi

    # Setup Next.js Frontend via Python installer
    # (handles Node.js auto-upgrade, npm install, and build in one place)
    echo ""
    echo "🎨 Setting up frontend..."
    if [ -f "install_frontend.py" ]; then
        if [ "$PULLED" = "true" ]; then
            $PYTHON_CMD install_frontend.py --force && FRONTEND_ENABLED=1 || FRONTEND_ENABLED=0
        else
            $PYTHON_CMD install_frontend.py && FRONTEND_ENABLED=1 || FRONTEND_ENABLED=0
        fi
    else
        echo "⚠️  install_frontend.py not found - skipping frontend setup"
        FRONTEND_ENABLED=0
    fi

    # Make startup scripts executable
    if [ -f "start_services_vastai.sh" ]; then
        chmod +x start_services_vastai.sh
    fi

    if [ -f "start_services_local.sh" ]; then
        chmod +x start_services_local.sh
    fi

    # Verify onnxruntime-gpu can see the GPU (requirements now use >=1.19 which
    # ships cuDNN 9 support from standard PyPI — no special index needed).
    if ! $PYTHON_CMD -c "import onnxruntime; print(onnxruntime.get_device())" 2>/dev/null | grep -q GPU; then
        echo "  onnxruntime-gpu doesn't see CUDA — attempting upgrade..."
        $PYTHON_CMD -m pip install --upgrade onnxruntime-gpu --no-cache-dir -q
    fi

    # Configure git for root usage
    # shellcheck disable=SC2046
    git config --global --add safe.directory $(pwd)

    # Create log directory
    mkdir -p /workspace/Ktiseos-Nyx-Trainer/logs

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
    echo "[$(date)] ERROR: Project directory not found at /workspace/Ktiseos-Nyx-Trainer" | tee -a /workspace/Ktiseos-Nyx-Trainer/logs/supervisor.log
    exit 1
fi
cd /workspace/Ktiseos-Nyx-Trainer

# VastAI's Caddy binds ports 3000/8000 and reverse-proxies to 13000/18000.
# Our services must listen on the proxy-target ports, not the Caddy-owned ports.
BACKEND_PORT="${BACKEND_PORT:-18000}"
FRONTEND_PORT="${FRONTEND_PORT:-13000}"
COMFYUI_PORT="${COMFYUI_PORT:-18188}"

# Clean up any existing processes on our ports (only if they're python/node processes)
# This prevents accidentally killing VastAI infrastructure (Caddy on 3000/8000)
echo "[$(date)] Checking for existing services on ports $BACKEND_PORT and $FRONTEND_PORT..." | tee -a /workspace/Ktiseos-Nyx-Trainer/logs/supervisor.log
pkill -f "uvicorn api.main:app" 2>/dev/null || true
pkill -f "next-server" 2>/dev/null || true
sleep 1

# Start backend
echo "[$(date)] Starting FastAPI backend on port $BACKEND_PORT..." | tee -a /workspace/Ktiseos-Nyx-Trainer/logs/supervisor.log
python -m uvicorn api.main:app --host 0.0.0.0 --port "$BACKEND_PORT" 2>&1 | tee -a /workspace/Ktiseos-Nyx-Trainer/logs/backend.log &
BACKEND_PID=$!

# Give backend a moment to start
sleep 2
EOL

    if [ "$FRONTEND_ENABLED" = "1" ]; then
        cat >> /opt/supervisor-scripts/ktiseos-nyx.sh << 'EOL'

# Start frontend (using custom server with WebSocket proxy)
echo "[$(date)] Starting Next.js frontend on port $FRONTEND_PORT..." | tee -a /workspace/Ktiseos-Nyx-Trainer/logs/supervisor.log
cd frontend || exit 1
PORT=$FRONTEND_PORT BACKEND_PORT=$BACKEND_PORT COMFYUI_PORT=$COMFYUI_PORT NODE_ENV=production node server.js 2>&1 | tee -a /workspace/Ktiseos-Nyx-Trainer/logs/frontend.log &
FRONTEND_PID=$!
EOL
    fi

    cat >> /opt/supervisor-scripts/ktiseos-nyx.sh << 'EOL'

# Wait for backend; also wait for frontend if it was started
wait $BACKEND_PID ${FRONTEND_PID:+$FRONTEND_PID}
EOL

    chmod +x /opt/supervisor-scripts/ktiseos-nyx.sh

    # ComfyUI gets its own supervisor process so backend/frontend restarts
    # don't kill it (and force a full model-reload).
    cat > /opt/supervisor-scripts/comfyui.sh << 'EOL'
#!/bin/bash
source /venv/main/bin/activate 2>/dev/null || true

COMFYUI_PORT="${COMFYUI_PORT:-18188}"

if [ ! -d /workspace/Ktiseos-Nyx-Trainer/ComfyUI ]; then
    echo "[$(date)] ComfyUI not installed — exiting."
    exit 0
fi

cd /workspace/Ktiseos-Nyx-Trainer
echo "[$(date)] Starting ComfyUI on port $COMFYUI_PORT..."
exec python ComfyUI/main.py --port "$COMFYUI_PORT" --listen 0.0.0.0 --enable-cors-header
EOL

    chmod +x /opt/supervisor-scripts/comfyui.sh

    cat > /etc/supervisor/conf.d/ktiseos-nyx.conf << 'EOL'
[program:ktiseos-nyx]
command=/opt/supervisor-scripts/ktiseos-nyx.sh
directory=/workspace/Ktiseos-Nyx-Trainer
autostart=true
autorestart=true
startsecs=10
stopasgroup=true
killasgroup=true
stdout_logfile=/workspace/Ktiseos-Nyx-Trainer/logs/supervisor.log
redirect_stderr=true
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=3

[program:comfyui]
command=/opt/supervisor-scripts/comfyui.sh
directory=/workspace/Ktiseos-Nyx-Trainer
autostart=true
autorestart=true
startsecs=30
stopasgroup=true
killasgroup=true
stdout_logfile=/workspace/Ktiseos-Nyx-Trainer/logs/comfyui.log
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
        if supervisorctl status comfyui | grep -q RUNNING; then
            echo "   ✅ comfyui started successfully!"
        else
            echo "   ⚠️  comfyui may not have started yet (takes ~30s to load)"
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
    echo "   - Backend:    /workspace/Ktiseos-Nyx-Trainer/logs/backend.log"
    echo "   - Frontend:   /workspace/Ktiseos-Nyx-Trainer/logs/frontend.log"
    echo "   - ComfyUI:    /workspace/Ktiseos-Nyx-Trainer/logs/comfyui.log"
    echo "   - Supervisor: /workspace/Ktiseos-Nyx-Trainer/logs/supervisor.log"
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
