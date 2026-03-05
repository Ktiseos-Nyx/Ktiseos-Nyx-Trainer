#!/bin/bash
# RunPod Provisioning Script for Ktiseos-Nyx-Trainer
#
# RunPod Template: Ktiseos-Nyx LoRA Trainer
#   Container Image: runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04
#   Exposed Ports: 8000/http, 3000/http, 22/tcp
#   Docker Command (auto-provisions on pod start, keeps Jupyter/SSH alive):
#     bash -c "/start.sh & sleep 5 && cd /workspace && git clone https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git 2>/dev/null; cd /workspace/Ktiseos-Nyx-Trainer && git pull && bash provision_runpod.sh"
#   Volume Mount Path: /workspace
#
# Manual usage (from Jupyter terminal, if not using Docker Command above):
#   cd /workspace && git clone https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git && cd Ktiseos-Nyx-Trainer && bash provision_runpod.sh
#
# On subsequent restarts (repo already cloned):
#   cd /workspace/Ktiseos-Nyx-Trainer && git pull && bash provision_runpod.sh
#
# Access URLs (once running):
#   Frontend: https://{POD_ID}-3000.proxy.runpod.net
#   Backend:  https://{POD_ID}-8000.proxy.runpod.net
#   API Docs: https://{POD_ID}-8000.proxy.runpod.net/docs
#   Jupyter:  https://{POD_ID}-8888.proxy.runpod.net (always available)

# Note: No 'set -e' for resilient provisioning - critical errors handled explicitly

provisioning_start() {
    echo "=========================================="
    echo "  Ktiseos-Nyx-Trainer Setup (RunPod)"
    echo "=========================================="

    # Log RunPod context
    [ -n "$RUNPOD_POD_ID" ] && echo "  Pod ID: $RUNPOD_POD_ID"
    [ -n "$RUNPOD_GPU_COUNT" ] && echo "  GPUs: $RUNPOD_GPU_COUNT"
    [ -n "$CUDA_VERSION" ] && echo "  CUDA: $CUDA_VERSION"
    echo ""

    # Ensure proper environment
    export PATH="$PATH:/usr/local/bin:/usr/bin:/bin"

    # Activate virtual environment
    # RunPod images may use different venv locations depending on the base image
    VENV_ACTIVATED=false
    for venv_path in "/venv/main/bin/activate" "/venv/bin/activate" "/workspace/venv/bin/activate"; do
        if [ -f "$venv_path" ]; then
            # shellcheck disable=SC1090
            source "$venv_path"
            echo "  Virtual environment activated: $venv_path"
            VENV_ACTIVATED=true
            break
        fi
    done
    if [ "$VENV_ACTIVATED" = false ]; then
        echo "  No venv found - using system Python"
    fi

    # Check for python
    if ! command -v python &> /dev/null; then
        if command -v python3 &> /dev/null; then
            echo "Creating python alias for python3..."
            alias python=python3
            export PATH="/usr/bin:$PATH"
        else
            echo "  Neither python nor python3 found!"
            exit 1
        fi
    fi

    # Handle git safe directory issue when running as root
    export GIT_CONFIG_GLOBAL=/tmp/temporary-git-config
    git config --file $GIT_CONFIG_GLOBAL --add safe.directory '*'

    # Navigate to workspace
    # RunPod persists /workspace across pod restarts (but not across pod deletions
    # unless using a network volume)
    if [ -d "/workspace/Ktiseos-Nyx-Trainer" ]; then
        echo "  Repository exists in /workspace, pulling latest changes..."
        # shellcheck disable=SC2164
        cd /workspace/Ktiseos-Nyx-Trainer
        git config --file $GIT_CONFIG_GLOBAL --add safe.directory "$(pwd)"
        git pull || echo "  Git pull failed, continuing with existing code"
    else
        echo "  Cloning repository..."
        # shellcheck disable=SC2164
        cd /workspace
        git clone https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
        # shellcheck disable=SC2164
        cd Ktiseos-Nyx-Trainer
    fi

    # Ensure we're using the right Python
    PYTHON_CMD=$(which python || which python3)
    echo "  Using Python: $PYTHON_CMD"

    # Check Node.js version (Next.js requires 18.18+)
    echo ""
    echo "  Checking Node.js installation..."
    SKIP_FRONTEND=false

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
            echo "  Node.js not found - installing via nvm..."
            curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
            export NVM_DIR="$HOME/.nvm"
            # shellcheck disable=SC1091
            [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
            nvm install 20
            nvm use 20
            if ! command -v node &> /dev/null; then
                echo "  Node.js installation failed! Frontend will be unavailable."
                SKIP_FRONTEND=true
            fi
        fi
    fi

    # Validate Node.js version if found
    if [ "$SKIP_FRONTEND" != true ] && command -v node &> /dev/null; then
        CURRENT_VERSION=$(node --version | sed 's/v//' | cut -d. -f1)
        if [ "$CURRENT_VERSION" -lt 18 ]; then
            echo "  Node.js $CURRENT_VERSION is too old (need 18+)"
            echo "   Frontend will be unavailable. Backend API will still work."
            SKIP_FRONTEND=true
        else
            echo "  Node.js ready: $(node --version) at $(which node)"
            echo "  npm version: $(npm --version)"
        fi
    fi

    # Run unified installer (handles all backend dependencies and setup)
    echo ""
    echo "  Running installer..."
    if [ -f "installer.py" ]; then
        $PYTHON_CMD installer.py
    else
        echo "  installer.py not found - falling back to manual dependency installation"
        $PYTHON_CMD -m pip install --upgrade pip -v

        # Skip PyTorch if already installed (common in RunPod base images)
        if $PYTHON_CMD -c "import torch" 2>/dev/null; then
            echo "  PyTorch already installed, skipping..."
        fi

        if [ -f "requirements_cloud.txt" ]; then
            $PYTHON_CMD -m pip install --upgrade setuptools wheel -v
            $PYTHON_CMD -m pip install -r requirements_cloud.txt --no-cache-dir -v
        elif [ -f "requirements.txt" ]; then
            echo "  Using requirements.txt"
            $PYTHON_CMD -m pip install -r requirements.txt --no-cache-dir -v
        fi
    fi

    # Setup Next.js Frontend
    if [ -d "frontend" ] && [ "$SKIP_FRONTEND" != true ]; then
        if command -v node &> /dev/null && command -v npm &> /dev/null; then
            echo ""
            echo "  Setting up Next.js frontend..."
            # shellcheck disable=SC2164
            cd frontend

            export NVM_DIR="$HOME/.nvm"
            # shellcheck disable=SC1091
            [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

            echo "   Installing npm packages..."
            # Remove stale lockfile from different OS to avoid platform-specific dep errors
            if [ -f "package-lock.json" ]; then
                echo "   Removing package-lock.json (regenerating for this platform)..."
                rm -f package-lock.json
            fi
            npm install --legacy-peer-deps || npm install --legacy-peer-deps --force || {
                echo "  npm install failed, continuing anyway..."
            }

            echo "   Building Next.js app..."
            npm run build || {
                echo "  Frontend build failed, services may not work correctly"
            }

            # shellcheck disable=SC2103
            cd ..
        else
            echo "  Node.js/npm not available - skipping frontend setup"
        fi
    elif [ "$SKIP_FRONTEND" = true ]; then
        echo "  Skipping frontend setup (Node.js not available)"
    else
        echo "  Frontend directory not found - skipping Next.js setup"
    fi

    # Configure git for root usage
    # shellcheck disable=SC2046
    git config --global --add safe.directory $(pwd)

    # Create log directory
    mkdir -p /workspace/logs

    # RunPod: Direct port binding (no Caddy reverse proxy like VastAI)
    # Services bind directly to the ports declared in the template
    BACKEND_PORT="${BACKEND_PORT:-8000}"
    FRONTEND_PORT="${FRONTEND_PORT:-3000}"

    echo ""
    echo "=========================================="
    echo "  Setup Complete!"
    echo "=========================================="
    echo ""
    echo "  Starting services..."
    echo "   Backend:  port $BACKEND_PORT"
    echo "   Frontend: port $FRONTEND_PORT"
    echo ""

    # Start backend
    echo "[$(date)] Starting FastAPI backend on port $BACKEND_PORT..." | tee -a /workspace/logs/backend.log
    $PYTHON_CMD -m uvicorn api.main:app --host 0.0.0.0 --port "$BACKEND_PORT" 2>&1 | tee -a /workspace/logs/backend.log &
    BACKEND_PID=$!

    sleep 2

    # Start frontend
    if [ -d "frontend/.next" ] && [ "$SKIP_FRONTEND" != true ]; then
        echo "[$(date)] Starting Next.js frontend on port $FRONTEND_PORT..." | tee -a /workspace/logs/frontend.log
        cd frontend || exit 1
        PORT=$FRONTEND_PORT BACKEND_PORT=$BACKEND_PORT NODE_ENV=production node server.js 2>&1 | tee -a /workspace/logs/frontend.log &
        FRONTEND_PID=$!
        cd ..
    else
        echo "  Frontend not available - running backend only"
    fi

    echo ""
    echo "=========================================="
    echo "  Services Running!"
    echo "=========================================="
    echo ""
    echo "  Access URLs:"
    if [ -n "$RUNPOD_POD_ID" ]; then
        echo "   Frontend: https://${RUNPOD_POD_ID}-${FRONTEND_PORT}.proxy.runpod.net"
        echo "   Backend:  https://${RUNPOD_POD_ID}-${BACKEND_PORT}.proxy.runpod.net"
        echo "   API Docs: https://${RUNPOD_POD_ID}-${BACKEND_PORT}.proxy.runpod.net/docs"
    else
        echo "   Frontend: http://0.0.0.0:${FRONTEND_PORT}"
        echo "   Backend:  http://0.0.0.0:${BACKEND_PORT}"
        echo "   API Docs: http://0.0.0.0:${BACKEND_PORT}/docs"
    fi
    echo ""
    echo "  Service logs:"
    echo "   Backend:  /workspace/logs/backend.log"
    echo "   Frontend: /workspace/logs/frontend.log"
    echo ""
    echo "  NOTE: RunPod HTTP proxy has a 100-second timeout."
    echo "  Long-running requests (training) use async job polling, so this is fine."
    echo ""

    # Keep container alive by waiting on child processes
    # If both die, tail the log so the container doesn't exit
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo "[$(date)] Services exited - tailing logs to keep container alive..."
    tail -f /workspace/logs/backend.log /workspace/logs/frontend.log 2>/dev/null &
    wait
}

# Check if provisioning should be skipped
if [[ ! -f /.noprovisioning ]]; then
    provisioning_start
fi
