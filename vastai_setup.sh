#!/bin/bash
# VastAI Provisioning Script for Ktiseos-Nyx-Trainer
# This script runs automatically when a VastAI instance starts
# Set via: PROVISIONING_SCRIPT=https://raw.githubusercontent.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/main/vastai_setup.sh

set -e  # Exit on error

echo "=========================================="
echo "🚀 Ktiseos-Nyx-Trainer Setup Starting..."
echo "=========================================="

# Activate VastAI's Python virtual environment
echo "📦 Activating Python environment..."
source /venv/main/bin/activate

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

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements-backend.txt

# Setup Kohya SD-Scripts (submodule)
echo "🔧 Setting up Kohya SD-Scripts..."
cd sd-scripts
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi
cd ..

# Check if frontend exists (they're migrating, so might not be there yet)
if [ -d "frontend" ]; then
    echo "🎨 Setting up Next.js frontend..."
    cd frontend

    # Use Node.js from NVM (VastAI base image has this)
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

    # Install dependencies
    npm install

    # Build for production
    echo "🏗️  Building Next.js app..."
    npm run build

    cd ..
else
    echo "⚠️  Frontend directory not found - skipping Next.js setup"
    echo "   (Expected during migration from Jupyter to Next.js)"
fi

# Make startup script executable
if [ -f "start_services.sh" ]; then
    chmod +x start_services.sh
fi

echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "🌐 Services will start automatically..."
echo "   Backend API: Port 8000"
echo "   Frontend UI: Port 3000"
echo ""
echo "Access your instance via the VastAI port forwarding URLs!"
echo ""
