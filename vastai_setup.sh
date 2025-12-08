#!/bin/bash
# VastAI Provisioning Script for Ktiseos-Nyx-Trainer
# This script runs automatically when a VastAI instance starts
# Set via: PROVISIONING_SCRIPT=https://raw.githubusercontent.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/main/vastai_setup.sh

set -e  # Exit on error

echo "=========================================="
echo "🚀 Ktiseos-Nyx-Trainer Setup Starting..."
echo "=========================================="

# Ensure proper environment setup for VastAI
export PATH="$PATH:/usr/local/bin:/usr/bin:/bin"

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

# Ensure we're using the right Python
PYTHON_CMD=$(which python || which python3)
echo "🐍 Using Python: $PYTHON_CMD"

# Install Node.js 20+ (needed for Next.js frontend and to resolve dependency compatibility issues)
echo "📦 Ensuring Node.js 20+ is installed..."
if command -v node &> /dev/null; then
    CURRENT_VERSION=$(node --version | sed 's/v//' | cut -d. -f1)
    if [ "$CURRENT_VERSION" -lt 20 ]; then
        echo "Current Node.js version ($CURRENT_VERSION) is too old, upgrading to Node.js 20+..."
    else
        echo "✅ Node.js version is sufficient: $(node --version)"
    fi
fi

# Install/upgrade to Node.js 20 regardless of what's currently there to avoid dependency conflicts
if command -v apt-get &> /dev/null; then
    # Try to install Node.js 20 using nodesource
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
else
    # Try using nvm to ensure Node.js 20
    export NVM_DIR="$HOME/.nvm"
    if [ -s "$NVM_DIR/nvm.sh" ]; then
        \. "$NVM_DIR/nvm.sh"
    else
        # Install nvm first, then install node 20
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
        export NVM_DIR="$HOME/.nvm"
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    fi
    nvm install 20
    nvm use 20
fi

echo "✅ Node.js installation complete: $(node --version)"
echo "✅ npm version: $(npm --version)"

# Run unified installer (handles all backend dependencies and setup)
echo "🔧 Running unified installer..."
if [ -f "installer.py" ]; then
    $PYTHON_CMD installer.py
else
    echo "⚠️  installer.py not found - falling back to manual dependency installation"

    # Fallback: Install dependencies manually
    echo "🐍 Installing all dependencies..."
    $PYTHON_CMD -m pip install --upgrade pip
    if [ -f "requirements.txt" ]; then
        # First try to resolve any version conflicts
        $PYTHON_CMD -m pip install --upgrade setuptools wheel
        # Install with conflict resolution
        $PYTHON_CMD -m pip install -r requirements.txt --no-cache-dir
    fi
fi

# Setup Next.js Frontend
if [ -d "frontend" ]; then
    echo "🎨 Setting up Next.js frontend..."
    cd frontend

    # Ensure Node.js is available
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

    # Check node version and install dependencies
    echo "Node version: $(node --version)"
    echo "NPM version: $(npm --version)"

    echo "   Installing npm packages..."
    npm ci --prefer-offline || npm install

    # Build for production
    echo "🏗️  Building Next.js app..."
    npm run build

    cd ..
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
    echo "   - Backend: $PYTHON_CMD -m uvicorn api.main:app --host 0.0.0.0 --port 8000"
    echo "   - Frontend: cd frontend && npm run start"

    # Start backend directly as fallback
    echo "🔧 Starting backend directly..."
    $PYTHON_CMD -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &

    # Start frontend directly as fallback
    if [ -d "frontend" ] && [ -d "frontend/.next" ]; then
        echo "🎨 Starting frontend directly..."
        cd frontend
        npm run start &
        cd ..
    fi

    # Keep the process running
    wait
fi
