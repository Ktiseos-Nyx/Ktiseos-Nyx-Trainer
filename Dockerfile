# Ktiseos Nyx LoRA Trainer - VastAI Template
# Optimized for CUDA 12.1+ with PyTorch 2.5.1
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    git-lfs \
    wget \
    curl \
    vim \
    nano \
    htop \
    tmux \
    screen \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip and install uv (ultra-fast package installer)
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install uv

# Set working directory
WORKDIR /workspace

# Install Node.js 20.x for frontend (Next.js)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g npm@latest \
    && rm -rf /var/lib/apt/lists/*

# Pre-install PyTorch with CUDA 12.1 support (saves time on container startup)
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Copy requirements files for layer caching
COPY requirements-api.txt /tmp/
COPY requirements-backend.txt /tmp/

# Install Python dependencies (backend + API)
# Use uv for faster installation
RUN uv pip install --system -r /tmp/requirements-backend.txt \
    && uv pip install --system -r /tmp/requirements-api.txt

# Copy application code
COPY api/ /workspace/Ktiseos-Nyx-Trainer/api/
COPY core/ /workspace/Ktiseos-Nyx-Trainer/core/
COPY widgets/ /workspace/Ktiseos-Nyx-Trainer/widgets/
COPY shared_managers.py /workspace/Ktiseos-Nyx-Trainer/
COPY installer.py /workspace/Ktiseos-Nyx-Trainer/

# Copy frontend code for building
COPY frontend/package*.json /workspace/Ktiseos-Nyx-Trainer/frontend/
COPY frontend/ /workspace/Ktiseos-Nyx-Trainer/frontend/

# Install frontend dependencies and build Next.js app
# This happens during Docker build, making startup much faster!
RUN cd /workspace/Ktiseos-Nyx-Trainer/frontend \
    && npm ci \
    && npm run build \
    && npm prune --production

# Create necessary directories
RUN mkdir -p /workspace/datasets \
    /workspace/output \
    /workspace/pretrained_model \
    /workspace/vae \
    /workspace/tagger_models \
    /workspace/training_config \
    /workspace/logs

# Expose ports
# 8000 - FastAPI Backend
# 3000 - Next.js Frontend
# 8888 - Jupyter (optional)
# 6006 - TensorBoard (optional)
EXPOSE 8000 3000 8888 6006

# Copy startup scripts
RUN chmod +x /startup.sh /start_services.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["/startup.sh"]
