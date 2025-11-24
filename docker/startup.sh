#!/bin/bash
# Ktiseos Nyx Trainer - Startup Script for VastAI/Docker
# This script handles initialization, dependency installation, and service startup

set -e  # Exit on error

echo "=================================================="
echo "  Ktiseos Nyx LoRA Trainer - Starting Up"
echo "=================================================="

# Navigate to workspace
cd /workspace/Ktiseos-Nyx-Trainer

# Repository code is already in the Docker image!
# Submodule logic is now handled by vendoring.

# Always run installer.py to ensure environment is validated and set up
echo "⚙️ Running unified installer.py to set up environment..."
python installer.py

# Python dependencies already installed in Docker image!
echo "Python dependencies already installed ✓"

# Frontend is pre-built in Docker image!
echo "Frontend already built during Docker image creation ✓"

# Start services
echo ""
echo "=================================================="
echo "  Starting Services"
echo "=================================================="

# Start FastAPI backend in background
echo "Starting FastAPI backend on port 8000..."
cd /workspace/Ktiseos-Nyx-Trainer
nohup uvicorn api.main:app --host 0.0.0.0 --port 8000 > /workspace/logs/backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend started (PID: $BACKEND_PID)"

# Wait for backend to be ready
echo "Waiting for backend to initialize..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Backend is ready!"
        break
    fi
    sleep 2
done

# Start Next.js frontend
echo "Starting Next.js frontend on port 3000..."
cd /workspace/Ktiseos-Nyx-Trainer/frontend
nohup npm run start > /workspace/logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend started (PID: $FRONTEND_PID)"

# Optional: Start Jupyter if requested
if [ "$START_JUPYTER" = "true" ]; then
    echo "Starting Jupyter Lab on port 8888..."
    cd /workspace/Ktiseos-Nyx-Trainer
    nohup jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
        --NotebookApp.token='' --NotebookApp.password='' \
        > /workspace/logs/jupyter.log 2>&1 &
    JUPYTER_PID=$!
    echo "Jupyter started (PID: $JUPYTER_PID)"
fi

# Optional: Start TensorBoard if requested
if [ "$START_TENSORBOARD" = "true" ]; then
    echo "Starting TensorBoard on port 6006..."
    nohup tensorboard --logdir=/workspace/training_logs --host=0.0.0.0 --port=6006 \
        > /workspace/logs/tensorboard.log 2>&1 &
    TENSORBOARD_PID=$!
    echo "TensorBoard started (PID: $TENSORBOARD_PID)"
fi

echo ""
echo "=================================================="
echo "  Ktiseos Nyx Trainer - Ready!"
echo "=================================================="
echo ""
echo "Services running:"
echo "  - Backend API:  http://localhost:8000"
echo "  - Frontend UI:  http://localhost:3000"
[ "$START_JUPYTER" = "true" ] && echo "  - Jupyter Lab:  http://localhost:8888"
[ "$START_TENSORBOARD" = "true" ] && echo "  - TensorBoard:  http://localhost:6006"
echo ""
echo "Logs available at: /workspace/logs/"
echo "Training outputs: /workspace/output/"
echo ""
echo "Press Ctrl+C to stop services"
echo "=================================================="

# Keep container running and monitor processes
while true; do
    # Check if backend is still running
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "ERROR: Backend process died! Check /workspace/logs/backend.log"
        exit 1
    fi

    # Check if frontend is still running
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "ERROR: Frontend process died! Check /workspace/logs/frontend.log"
        exit 1
    fi

    sleep 10
done
