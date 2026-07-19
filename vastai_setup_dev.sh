#!/bin/bash
# VastAI Provisioning Script — Ecosystem DEV BRANCH
#
# ⚠️  BETA TESTERS ONLY — pulls from the dev branch, not main.
#     Expect rough edges, breaking changes, and more frequent updates.
#
# Set via:
#   PROVISIONING_SCRIPT=https://raw.githubusercontent.com/UselessToys/Ecosystem_WebUI/dev/vastai_setup_dev.sh

# Note: No 'set -e' for resilient provisioning — critical errors handled explicitly

provisioning_start() {
    echo "=========================================="
    echo "🧪 Ecosystem DEV Setup Starting"
    echo "   ⚠️  BETA — pulling from 'dev' branch"
    echo "=========================================="

    export PATH="$PATH:/usr/local/bin:/usr/bin:/bin"

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

    export GIT_CONFIG_GLOBAL=/tmp/temporary-git-config
    git config --file $GIT_CONFIG_GLOBAL --add safe.directory '*'

    if [ -d "/opt/workspace-internal/Ecosystem_WebUI" ]; then
        echo "📂 Using pre-installed code from Docker image..."
        # shellcheck disable=SC2164
        cd /opt/workspace-internal/Ecosystem_WebUI
        git config --file $GIT_CONFIG_GLOBAL --add safe.directory "$(pwd)"
        git fetch origin dev 2>&1
        PULL_OUTPUT=$(git checkout dev && git pull origin dev 2>&1)
        PULL_EXIT=$?
        echo "$PULL_OUTPUT"
        if [ $PULL_EXIT -ne 0 ]; then
            echo "⚠️ Git pull failed, continuing with existing code"
        elif ! echo "$PULL_OUTPUT" | grep -q "Already up to date"; then
            PULLED=true
        fi
    elif [ -d "/workspace/Ecosystem_WebUI" ]; then
        echo "📂 Repository exists in /workspace, pulling latest dev changes..."
        # shellcheck disable=SC2164
        cd /workspace/Ecosystem_WebUI
        git config --file $GIT_CONFIG_GLOBAL --add safe.directory "$(pwd)"
        git fetch origin dev 2>&1
        PULL_OUTPUT=$(git checkout dev && git pull origin dev 2>&1)
        PULL_EXIT=$?
        echo "$PULL_OUTPUT"
        if [ $PULL_EXIT -ne 0 ]; then
            echo "⚠️ Git pull failed, continuing with existing code"
        elif ! echo "$PULL_OUTPUT" | grep -q "Already up to date"; then
            PULLED=true
        fi
    else
        echo "📥 Cloning repository (dev branch)..."
        # shellcheck disable=SC2164
        cd /workspace
        git clone --branch dev https://github.com/UselessToys/Ecosystem_WebUI.git
        # shellcheck disable=SC2164
        cd Ecosystem_WebUI
    fi

    echo "🌿 Branch: $(git branch --show-current)"

    PYTHON_CMD=$(which python || which python3)
    echo "🐍 Using Python: $PYTHON_CMD"

    echo "🔧 Running backend installer..."
    if [ -f "installer.py" ]; then
        $PYTHON_CMD installer.py
    else
        echo "⚠️  installer.py not found - falling back to manual dependency installation"
        $PYTHON_CMD -m pip install --upgrade pip -v
        if [ -f "requirements_cloud.txt" ]; then
            $PYTHON_CMD -m pip install --upgrade setuptools wheel -v
            $PYTHON_CMD -m pip install -r requirements_cloud.txt --no-cache-dir -v
        elif [ -f "requirements.txt" ]; then
            echo "⚠️  requirements_cloud.txt not found, using requirements.txt"
            $PYTHON_CMD -m pip install -r requirements.txt --no-cache-dir -v
        fi
    fi

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

    if [ -f "start_services_vastai.sh" ]; then
        chmod +x start_services_vastai.sh
    fi

    if [ -f "start_services_local.sh" ]; then
        chmod +x start_services_local.sh
    fi

    if ! $PYTHON_CMD -c "import onnxruntime; print(onnxruntime.get_device())" 2>/dev/null | grep -q GPU; then
        echo "  onnxruntime-gpu doesn't see CUDA — attempting upgrade..."
        $PYTHON_CMD -m pip install --upgrade onnxruntime-gpu --no-cache-dir -q
    fi

    # shellcheck disable=SC2046
    git config --global --add safe.directory $(pwd)

    mkdir -p /workspace/Ecosystem_WebUI/logs

    echo ""
    echo "📝 Creating supervisor config for auto-restart..."

    mkdir -p /opt/supervisor-scripts

    cat > /opt/supervisor-scripts/ktiseos-nyx.sh << 'EOL'
#!/bin/bash
source /venv/main/bin/activate 2>/dev/null || true

if ! command -v node &> /dev/null; then
    for node_path in /opt/nvm/versions/node/*/bin /usr/bin /usr/local/bin; do
        if [ -f "$node_path/node" ]; then
            export PATH="$node_path:$PATH"
            break
        fi
    done
fi

if [ ! -d /workspace/Ecosystem_WebUI ]; then
    echo "[$(date)] ERROR: Project directory not found at /workspace/Ecosystem_WebUI" | tee -a /workspace/Ecosystem_WebUI/logs/supervisor.log
    exit 1
fi
cd /workspace/Ecosystem_WebUI

BACKEND_PORT="${BACKEND_PORT:-18000}"
FRONTEND_PORT="${FRONTEND_PORT:-13000}"
COMFYUI_PORT="${COMFYUI_PORT:-18188}"

echo "[$(date)] Checking for existing services on ports $BACKEND_PORT and $FRONTEND_PORT..." | tee -a /workspace/Ecosystem_WebUI/logs/supervisor.log
pkill -f "uvicorn api.main:app" 2>/dev/null || true
pkill -f "next-server" 2>/dev/null || true
sleep 1

echo "[$(date)] Starting FastAPI backend on port $BACKEND_PORT..." | tee -a /workspace/Ecosystem_WebUI/logs/supervisor.log
python -m uvicorn api.main:app --host 0.0.0.0 --port "$BACKEND_PORT" 2>&1 | tee -a /workspace/Ecosystem_WebUI/logs/backend.log &
BACKEND_PID=$!

sleep 2
EOL

    if [ "$FRONTEND_ENABLED" = "1" ]; then
        cat >> /opt/supervisor-scripts/ktiseos-nyx.sh << 'EOL'

echo "[$(date)] Starting Next.js frontend on port $FRONTEND_PORT..." | tee -a /workspace/Ecosystem_WebUI/logs/supervisor.log
cd frontend || exit 1
PORT=$FRONTEND_PORT BACKEND_PORT=$BACKEND_PORT COMFYUI_PORT=$COMFYUI_PORT NODE_ENV=production node server.js 2>&1 | tee -a /workspace/Ecosystem_WebUI/logs/frontend.log &
FRONTEND_PID=$!
EOL
    fi

    cat >> /opt/supervisor-scripts/ktiseos-nyx.sh << 'EOL'

wait $BACKEND_PID ${FRONTEND_PID:+$FRONTEND_PID}
EOL

    chmod +x /opt/supervisor-scripts/ktiseos-nyx.sh

    # ComfyUI gets its own supervisor process so backend/frontend restarts
    # don't kill it (and force a full model-reload).
    cat > /opt/supervisor-scripts/comfyui.sh << 'EOL'
#!/bin/bash
source /venv/main/bin/activate 2>/dev/null || true

COMFYUI_PORT="${COMFYUI_PORT:-18188}"

if [ ! -d /workspace/Ecosystem_WebUI/ComfyUI ]; then
    echo "[$(date)] ComfyUI not installed — exiting."
    exit 0
fi

cd /workspace/Ecosystem_WebUI
echo "[$(date)] Starting ComfyUI on port $COMFYUI_PORT..."
exec python ComfyUI/main.py --port "$COMFYUI_PORT" --listen 0.0.0.0 --enable-cors-header
EOL

    chmod +x /opt/supervisor-scripts/comfyui.sh

    cat > /etc/supervisor/conf.d/ktiseos-nyx.conf << 'EOL'
[program:ktiseos-nyx]
command=/opt/supervisor-scripts/ktiseos-nyx.sh
directory=/workspace/Ecosystem_WebUI
autostart=true
autorestart=true
startsecs=10
stopasgroup=true
killasgroup=true
stdout_logfile=/workspace/Ecosystem_WebUI/logs/supervisor.log
redirect_stderr=true
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=3

[program:comfyui]
command=/opt/supervisor-scripts/comfyui.sh
directory=/workspace/Ecosystem_WebUI
autostart=true
autorestart=true
startsecs=30
stopasgroup=true
killasgroup=true
stdout_logfile=/workspace/Ecosystem_WebUI/logs/comfyui.log
redirect_stderr=true
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=3
EOL

    echo "   ✅ Supervisor config created"

    echo ""
    echo "=========================================="
    echo "✅ DEV Setup Complete!"
    echo "=========================================="
    echo ""
    echo "🌿 Running branch: $(git branch --show-current)"
    echo "🚀 Activating supervisor to start services..."

    if command -v supervisorctl &> /dev/null; then
        supervisorctl reread
        supervisorctl update
        sleep 5
        if supervisorctl status ktiseos-nyx | grep -q RUNNING; then
            echo "   ✅ ktiseos-nyx started successfully!"
        else
            echo "   ⚠️  ktiseos-nyx may not have started - check logs"
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
    echo "   - Frontend: Next.js UI (port ${FRONTEND_PORT:-13000})"
    echo "   - Backend:  FastAPI (port ${BACKEND_PORT:-18000})"
    echo "   - ComfyUI:  (port ${COMFYUI_PORT:-18188})"
    echo ""
    echo "♻️  Auto-restart is enabled (ComfyUI restarts independently of backend/frontend)!"
    echo ""
    echo "📋 Service logs:"
    echo "   - Backend:    /workspace/Ecosystem_WebUI/logs/backend.log"
    echo "   - Frontend:   /workspace/Ecosystem_WebUI/logs/frontend.log"
    echo "   - ComfyUI:    /workspace/Ecosystem_WebUI/logs/comfyui.log"
    echo "   - Supervisor: /workspace/Ecosystem_WebUI/logs/supervisor.log"
    echo ""
    echo "🔧 Manual service control:"
    echo "   - Restart all:    supervisorctl restart all"
    echo "   - Restart app:    supervisorctl restart ktiseos-nyx"
    echo "   - Restart ComfyUI: supervisorctl restart comfyui"
    echo "   - Status:         supervisorctl status"
    echo ""
    echo "🔄 Pull latest dev changes:"
    echo "   cd /workspace/Ecosystem_WebUI && git pull origin dev"
    echo "   python install_frontend.py --force"
    echo "   supervisorctl restart ktiseos-nyx"
    echo ""
}

if [[ ! -f /.noprovisioning ]]; then
    provisioning_start
fi
