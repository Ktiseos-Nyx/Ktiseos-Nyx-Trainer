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
    git clone https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
else
    echo "📂 Repository already exists, pulling latest changes..."
    cd Ktiseos-Nyx-Trainer
    git pull
    cd ..
fi

cd Ktiseos-Nyx-Trainer

# Run unified installer (handles all backend dependencies and setup)
echo "🔧 Running unified installer..."
if [ -f "installer.py" ]; then
    python installer.py
else
    echo "⚠️  installer.py not found - falling back to manual dependency installation"

    # Fallback: Install dependencies manually
    echo "🐍 Installing backend dependencies..."
    pip install --upgrade pip
    if [ -f "requirements-backend.txt" ]; then
        pip install -r requirements-backend.txt
    fi

    # Install API dependencies
    echo "🔌 Installing API dependencies..."
    if [ -f "requirements-api.txt" ]; then
        pip install -r requirements-api.txt
    fi
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
if [ -f "start_services_vastai.sh" ]; then
    chmod +x start_services_vastai.sh
fi

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "🚀 Starting services..."
echo ""

# Start the services
if [ -f "start_services_vastai.sh" ]; then
    ./start_services_vastai.sh
else
    echo "⚠️  start_services_vastai.sh not found"
    echo "   Manually start services with:"
    echo "   - Backend: uvicorn api.main:app --host 0.0.0.0 --port 8000"
    echo "   - Frontend: cd frontend && npm run start"
fi
