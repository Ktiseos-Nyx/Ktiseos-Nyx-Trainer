#!/bin/bash
# VastAI Provisioning Script for Ktiseos-Nyx-Trainer
# This script runs automatically when a VastAI instance starts
# Set via: PROVISIONING_SCRIPT=https://raw.githubusercontent.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/main/vastai_setup.sh

set -e  # Exit on error

echo "=========================================="
echo "🚀 Ktiseos-Nyx-Trainer Setup Starting..."
echo "=========================================="

# Note: VastAI PyTorch template already has venv activated
# No need to activate it manually!

# Navigate to workspace
cd /workspace

# Clone the repository if it doesn't exist
if [ ! -d "Ktiseos-Nyx-Trainer" ]; then
    echo "📥 Cloning repository..."
    git clone --recursive https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
else
    echo "📂 Repository already exists, pulling latest changes..."
    cd Ktiseos-Nyx-Trainer
    git pull
    git submodule update --init --recursive
    cd ..
fi

cd Ktiseos-Nyx-Trainer

# Install Python dependencies (backend)
echo "🐍 Installing backend dependencies..."
pip install --upgrade pip
if [ -f "requirements-backend.txt" ]; then
    pip install -r requirements-backend.txt
fi

# Install Python dependencies (API)
echo "🔌 Installing API dependencies..."
if [ -f "requirements-api.txt" ]; then
    pip install -r requirements-api.txt
fi

# Setup Derrian Backend with SD-Scripts (submodule)
echo "🔧 Setting up Derrian Backend & SD-Scripts..."
if [ -d "trainer/derrian_backend" ]; then
    cd trainer/derrian_backend

    # Run backend installer with piped answers (non-interactive)
    # Question 1: "are you using this remotely?" -> y (VastAI is remote)
    # Question 2: "do you want to use ngrok?" -> n (we use VastAI port forwarding)
    if [ -f "installer.py" ]; then
        echo "   Running Derrian backend installer (non-interactive)..."
        echo -e "y\nn" | python installer.py
    fi

    # Install sd-scripts dependencies
    if [ -d "sd_scripts" ]; then
        cd sd_scripts
        if [ -f "requirements.txt" ]; then
            echo "   Installing sd-scripts requirements..."
            pip install -r requirements.txt
        fi
        cd ..
    fi

    cd ../..
else
    echo "⚠️  Derrian backend not found - skipping sd-scripts setup"
fi

# Setup Next.js Frontend
if [ -d "frontend" ]; then
    echo "🎨 Setting up Next.js frontend..."
    cd frontend

    # Use Node.js from NVM (VastAI PyTorch base image has this)
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

    # Install dependencies
    echo "   Installing npm packages..."
    npm install

    # Build for production
    echo "🏗️  Building Next.js app..."
    npm run build

    cd ..
else
    echo "⚠️  Frontend directory not found - skipping Next.js setup"
fi

# Make startup script executable
if [ -f "start_services.sh" ]; then
    chmod +x start_services.sh
fi

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "🚀 Starting services..."
echo ""

# Start the services
if [ -f "start_services.sh" ]; then
    ./start_services.sh
else
    echo "⚠️  start_services.sh not found"
    echo "   Manually start services with:"
    echo "   - Backend: python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"
    echo "   - Frontend: cd frontend && npm start -- -p 3000"
fi
