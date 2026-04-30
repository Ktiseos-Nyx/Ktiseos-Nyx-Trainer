#!/bin/bash
# RunPod Provisioning Script — Ktiseos-Nyx-Trainer DEV BRANCH
#
# ⚠️  BETA TESTERS ONLY — pulls from the dev branch, not main.
#     Expect rough edges, breaking changes, and more frequent updates.
#
# RunPod Template: Ktiseos-Nyx LoRA Trainer (DEV)
#   Container Image: runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04
#   HTTP Ports: 8888, 6006, 3000, 8000
#   TCP Ports: 22
#   Docker Command:
#     bash -c "/start.sh & sleep 5 && cd /workspace && git clone --branch dev https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git 2>/dev/null; cd /workspace/Ktiseos-Nyx-Trainer && git checkout dev && git pull origin dev && bash provision_runpod_dev.sh"
#   Volume Mount Path: /workspace
#
# Manual usage:
#   cd /workspace && git clone --branch dev https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git && cd Ktiseos-Nyx-Trainer && bash provision_runpod_dev.sh
#
# On subsequent restarts:
#   cd /workspace/Ktiseos-Nyx-Trainer && git checkout dev && git pull origin dev && bash provision_runpod_dev.sh
#
# Access URLs (once running):
#   Frontend: https://{POD_ID}-3000.proxy.runpod.net
#   Backend:  https://{POD_ID}-8000.proxy.runpod.net
#   Jupyter:  https://{POD_ID}-8888.proxy.runpod.net

# Note: No 'set -e' for resilient provisioning — critical errors handled explicitly

provisioning_start() {
    echo "=========================================="
    echo "  Ktiseos-Nyx-Trainer DEV Setup (RunPod)"
    echo "  ⚠️  BETA — pulling from 'dev' branch"
    echo "=========================================="

    [ -n "$RUNPOD_POD_ID" ] && echo "  Pod ID: $RUNPOD_POD_ID"
    [ -n "$RUNPOD_GPU_COUNT" ] && echo "  GPUs: $RUNPOD_GPU_COUNT"
    [ -n "$CUDA_VERSION" ] && echo "  CUDA: $CUDA_VERSION"
    echo ""

    export PATH="$PATH:/usr/local/bin:/usr/bin:/bin"

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

    if ! command -v python &> /dev/null; then
        if command -v python3 &> /dev/null; then
            echo "Symlinking python3 → python..."
            ln -sf "$(command -v python3)" /usr/local/bin/python
        else
            echo "  Neither python nor python3 found!"
            exit 1
        fi
    fi

    export GIT_CONFIG_GLOBAL=/tmp/temporary-git-config
    git config --file $GIT_CONFIG_GLOBAL --add safe.directory '*'

    if [ -d "/workspace/Ktiseos-Nyx-Trainer" ]; then
        echo "  Repository exists in /workspace, pulling latest dev changes..."
        # shellcheck disable=SC2164
        cd /workspace/Ktiseos-Nyx-Trainer
        git config --file $GIT_CONFIG_GLOBAL --add safe.directory "$(pwd)"
        git fetch origin dev 2>&1
        PULL_OUTPUT=$(git checkout dev && git pull origin dev 2>&1)
        PULL_EXIT=$?
        echo "$PULL_OUTPUT"
        if [ $PULL_EXIT -ne 0 ]; then
            echo "  Git pull failed, continuing with existing code"
        elif ! echo "$PULL_OUTPUT" | grep -q "Already up to date"; then
            PULLED=true
        fi
    else
        echo "  Cloning repository (dev branch)..."
        # shellcheck disable=SC2164
        cd /workspace
        git clone --branch dev https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
        # shellcheck disable=SC2164
        cd Ktiseos-Nyx-Trainer
    fi

    echo "  Branch: $(git branch --show-current)"

    PYTHON_CMD=$(which python || which python3)
    echo "  Using Python: $PYTHON_CMD"

    echo ""
    echo "  Running backend installer..."
    if [ -f "installer.py" ]; then
        $PYTHON_CMD installer.py
    else
        echo "  installer.py not found - falling back to manual dependency installation"
        $PYTHON_CMD -m pip install --upgrade pip -v

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

    echo ""
    echo "  Setting up frontend..."
    if [ -f "install_frontend.py" ]; then
        if [ "$PULLED" = "true" ]; then
            $PYTHON_CMD install_frontend.py --force
        else
            $PYTHON_CMD install_frontend.py
        fi
    else
        echo "  install_frontend.py not found - skipping frontend setup"
        SKIP_FRONTEND=true
    fi

    if ! $PYTHON_CMD -c "import onnxruntime; print(onnxruntime.get_device())" 2>/dev/null | grep -q GPU; then
        echo "  onnxruntime-gpu doesn't see CUDA — attempting upgrade..."
        $PYTHON_CMD -m pip install --upgrade onnxruntime-gpu --no-cache-dir -q
    fi

    # shellcheck disable=SC2046
    git config --global --add safe.directory $(pwd)

    mkdir -p /workspace/logs

    BACKEND_PORT="${BACKEND_PORT:-8000}"
    FRONTEND_PORT="${FRONTEND_PORT:-3000}"

    echo ""
    echo "=========================================="
    echo "  DEV Setup Complete!"
    echo "=========================================="
    echo ""
    echo "  Branch: $(git branch --show-current)"
    echo "  Starting services..."
    echo "   Backend:  port $BACKEND_PORT"
    echo "   Frontend: port $FRONTEND_PORT"
    echo ""

    echo "[$(date)] Starting FastAPI backend on port $BACKEND_PORT..." | tee -a /workspace/logs/backend.log
    $PYTHON_CMD -m uvicorn api.main:app --host 0.0.0.0 --port "$BACKEND_PORT" 2>&1 | tee -a /workspace/logs/backend.log &
    BACKEND_PID=$!

    sleep 2

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
    fi
    echo ""
    echo "  Service logs:"
    echo "   Backend:  /workspace/logs/backend.log"
    echo "   Frontend: /workspace/logs/frontend.log"
    echo ""
    echo "  NOTE: RunPod HTTP proxy has a 100-second timeout."
    echo "  Long-running requests (training) use async job polling, so this is fine."
    echo ""
    echo "  🔄 Pull latest dev changes:"
    echo "   cd /workspace/Ktiseos-Nyx-Trainer && git pull origin dev"
    echo "   python install_frontend.py --force && bash provision_runpod_dev.sh"
    echo ""

    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo "[$(date)] Services exited - tailing logs to keep container alive..."
    tail -f /workspace/logs/backend.log /workspace/logs/frontend.log 2>/dev/null &
    wait
}

if [[ ! -f /.noprovisioning ]]; then
    provisioning_start
fi
