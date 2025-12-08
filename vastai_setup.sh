#!/bin/bash
# VastAI Provisioning Script for Ktiseos-Nyx-Trainer
# This script runs automatically when a VastAI instance starts
# Set via: PROVISIONING_SCRIPT=https://raw.githubusercontent.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/main/vastai_setup.sh

set -e  # Exit on error

provisioning_start() {
    echo "=========================================="
    echo "🚀 Ktiseos-Nyx-Trainer Setup Starting..."
    echo "=========================================="

    # Ensure proper environment setup for VastAI
    export PATH="$PATH:/usr/local/bin:/usr/bin:/bin"

    # Activate virtual environment (VastAI PyTorch base image has this)
    if [ -f "/venv/main/bin/activate" ]; then
        source /venv/main/bin/activate
        echo "✅ Virtual environment activated"
    fi

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
    cd /workspace

    # Clone the repository if it doesn't exist
    if [ ! -d "Ktiseos-Nyx-Trainer" ]; then
        echo "📥 Cloning repository..."
        git clone https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
    else
        echo "📂 Repository exists, pulling latest changes..."
        cd Ktiseos-Nyx-Trainer
        git config --file $GIT_CONFIG_GLOBAL --add safe.directory "$(pwd)"
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
        apt-get install -y nodejs npm
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

    # Configure git for root usage (similar to SD-Forge script)
    git config --global --add safe.directory $(pwd)

    # Set up the instance portal configuration for VastAI
    echo "🔧 Configuring Instance Portal..."
    rm -f /etc/portal.yaml
    export PORTAL_CONFIG="localhost:3000:3000:/:NextJS Frontend|localhost:8000:8000:/:FastAPI Backend|localhost:8888:8888:/:Jupyter|localhost:6006:6006:/:TensorBoard"

    # Create Supervisor startup script for our services
    cat > /opt/supervisor-scripts/ktiseos-nyx.sh << 'EOL'
#!/bin/bash

kill_subprocesses() {
    local pid=$1
    local subprocesses=$(pgrep -P "$pid")

    for process in $subprocesses; do
        kill_subprocesses "$process"
    done

    if [[ -n "$subprocesses" ]]; then
        kill -TERM $subprocesses 2>/dev/null
    fi
}

cleanup() {
    kill_subprocesses $$
    sleep 2
    pkill -KILL -P $$ 2>/dev/null
    exit 0
}

trap cleanup EXIT INT TERM

# Wait for portal config and check if our services should be started
while [ ! -f "$(realpath -q /etc/portal.yaml 2>/dev/null)" ]; do
    echo "Waiting for /etc/portal.yaml before starting ${PROC_NAME}..." | tee -a "/var/log/portal/${PROC_NAME}.log"
    sleep 1
done

# Check for our services in the portal config
search_term="NextJS Frontend"
search_pattern=$(echo "$search_term" | sed 's/[ _-]/[ _-]/g')
if ! grep -qiE "^[^#].*${search_pattern}" /etc/portal.yaml; then
    echo "Skipping startup for ${PROC_NAME} (not in /etc/portal.yaml)" | tee -a "/var/log/portal/${PROC_NAME}.log"
    exit 0
fi

echo "Starting Ktiseos Nyx Trainer services" | tee "/var/log/portal/${PROC_NAME}.log"

# Activate virtual environment
. /venv/main/bin/activate

# Navigate to the project directory
cd /workspace/Ktiseos-Nyx-Trainer

# Clean up any existing processes on our ports
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

# Start the backend first
echo "Starting FastAPI backend..." | tee -a "/var/log/portal/${PROC_NAME}.log"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Start the frontend
echo "Starting NextJS frontend..." | tee -a "/var/log/portal/${PROC_NAME}.log"
cd frontend && npm run start &
FRONTEND_PID=$!

# Keep the script running and monitor processes
wait $BACKEND_PID $FRONTEND_PID 2>/dev/null

EOL

    chmod +x /opt/supervisor-scripts/ktiseos-nyx.sh

    # Generate the supervisor config file
    cat > /etc/supervisor/conf.d/ktiseos-nyx.conf << 'EOL'
[program:ktiseos-nyx]
environment=PROC_NAME="%(program_name)s",WORKSPACE="/workspace"
command=/opt/supervisor-scripts/ktiseos-nyx.sh
directory=/workspace/Ktiseos-Nyx-Trainer
autostart=true
autorestart=true
exitcodes=0
startsecs=0
stopasgroup=true
killasgroup=true
stopsignal=TERM
stopwaitsecs=10
# This is necessary for Vast logging to work alongside the Portal logs (Must output to /dev/stdout)
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_events_enabled=true
stdout_logfile_maxbytes=0
stdout_logfile_backups=0
EOL

    # Reload Supervisor to apply new configuration
    if command -v supervisorctl &> /dev/null; then
        supervisorctl reread
        supervisorctl update
    fi

    echo ""
    echo "=========================================="
    echo "✅ Setup Complete!"
    echo "=========================================="
    echo ""
    echo "🚀 Services will start automatically via VastAI's supervisor..."
    echo ""
    echo "🌐 Access your applications via the portal links on your instance page:"
    echo "   - Frontend: NextJS UI (port 3000)"
    echo "   - Backend: FastAPI API (port 8000)"
    echo "   - Jupyter: For diagnostics (port 8888)"
    echo "   - TensorBoard: For monitoring (port 6006)"
    echo ""
    echo "ℹ️  Supervisor will manage services with proper logging and restart capabilities."
    echo ""
}

# Check if provisioning should be skipped
if [[ ! -f /.noprovisioning ]]; then
    provisioning_start
fi
