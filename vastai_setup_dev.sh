#!/bin/bash
# VastAI Provisioning Script — Ktiseos-Nyx-Trainer DEV BRANCH
#
# ⚠️  BETA TESTERS ONLY — pulls from the dev branch, not main.
#     Expect rough edges, breaking changes, and more frequent updates.
#
# Set via:
#   PROVISIONING_SCRIPT=https://raw.githubusercontent.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/dev/vastai_setup_dev.sh

# Note: No 'set -e' for resilient provisioning — critical errors handled explicitly

provisioning_start() {
    echo "=========================================="
    echo "🧪 Ktiseos-Nyx-Trainer DEV Setup Starting"
    echo "   ⚠️  BETA — pulling from 'dev' branch"
    echo "=========================================="

    export PATH="$PATH:/usr/local/bin:/usr/bin:/bin"

    if [ -f "/venv/main/bin/activate" ]; then
        # shellcheck disable=SC1091
        source /venv/main/bin/activate
        echo "✅ Virtual environment activated"
    fi

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

    if [ -d "/opt/workspace-internal/Ktiseos-Nyx-Trainer" ]; then
        echo "📂 Using pre-installed code from Docker image..."
        # shellcheck disable=SC2164
        cd /opt/workspace-internal/Ktiseos-Nyx-Trainer
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
    elif [ -d "/workspace/Ktiseos-Nyx-Trainer" ]; then
        echo "📂 Repository exists in /workspace, pulling latest dev changes..."
        # shellcheck disable=SC2164
        cd /workspace/Ktiseos-Nyx-Trainer
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
        git clone --branch dev https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
        # shellcheck disable=SC2164
        cd Ktiseos-Nyx-Trainer
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

    mkdir -p /workspace/logs

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

if [ ! -d /workspace/Ktiseos-Nyx-Trainer ]; then
    echo "[$(date)] ERROR: Project directory not found at /workspace/Ktiseos-Nyx-Trainer" | tee -a /workspace/logs/supervisor.log
    exit 1
fi
cd /workspace/Ktiseos-Nyx-Trainer

BACKEND_PORT="${BACKEND_PORT:-18000}"
FRONTEND_PORT="${FRONTEND_PORT:-13000}"

echo "[$(date)] Checking for existing services on ports $BACKEND_PORT and $FRONTEND_PORT..." | tee -a /workspace/logs/supervisor.log
pkill -f "uvicorn api.main:app" 2>/dev/null || true
pkill -f "next-server" 2>/dev/null || true
sleep 1

echo "[$(date)] Starting FastAPI backend on port $BACKEND_PORT..." | tee -a /workspace/logs/supervisor.log
python -m uvicorn api.main:app --host 0.0.0.0 --port "$BACKEND_PORT" 2>&1 | tee -a /workspace/logs/backend.log &
BACKEND_PID=$!

sleep 2
EOL

    if [ "$FRONTEND_ENABLED" = "1" ]; then
        cat >> /opt/supervisor-scripts/ktiseos-nyx.sh << 'EOL'

echo "[$(date)] Starting Next.js frontend on port $FRONTEND_PORT..." | tee -a /workspace/logs/supervisor.log
cd frontend || exit 1
PORT=$FRONTEND_PORT BACKEND_PORT=$BACKEND_PORT NODE_ENV=production node server.js 2>&1 | tee -a /workspace/logs/frontend.log &
FRONTEND_PID=$!
EOL
    fi

    cat >> /opt/supervisor-scripts/ktiseos-nyx.sh << 'EOL'

wait $BACKEND_PID ${FRONTEND_PID:+$FRONTEND_PID}
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
    echo "   - Backend:  FastAPI (port 8000)"
    echo ""
    echo "♻️  Auto-restart is enabled!"
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
    echo "🔄 Pull latest dev changes:"
    echo "   cd /workspace/Ktiseos-Nyx-Trainer && git pull origin dev"
    echo "   python install_frontend.py --force"
    echo "   supervisorctl restart ktiseos-nyx"
    echo ""
}

if [[ ! -f /.noprovisioning ]]; then
    provisioning_start
fi
