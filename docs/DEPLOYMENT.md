# Ktiseos Nyx LoRA Trainer - Deployment Guide

> ⚠️ **BETA - IN ACTIVE DEVELOPMENT**
>
> This project is currently in beta and under active development. Features may not work as expected, and breaking changes may occur between versions. Please test thoroughly before using in production and report any issues you encounter.
>
> **Report issues**: https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues

Complete deployment guide for VastAI, local Docker, and development environments.

---

## 🚀 VastAI Deployment (Recommended for Training)

VastAI provides affordable GPU instances perfect for LoRA training.

### Quick Start

1. **Create VastAI Account**: Sign up at [vast.ai](https://vast.ai)

2. **Search for GPU Instance**:
   - Minimum: 12GB VRAM (RTX 3060, RTX 4060 Ti)
   - Recommended: 24GB VRAM (RTX 3090, RTX 4090, A5000)
   - CUDA Version: 12.1 or higher

3. **Deploy Template**:
   ```bash
   # Option 1: Use pre-built template from VastAI marketplace
   # Search for "Ktiseos Nyx LoRA Trainer"

   # Option 2: Deploy from GitHub
   # Select "Custom Docker Image" and provide GitHub repo URL
   ```

4. **Access Services**:
   - **Web UI**: `http://<instance-ip>:3000`
   - **API**: `http://<instance-ip>:8000`
   - **Jupyter** (optional): `http://<instance-ip>:8888`
   - **TensorBoard** (optional): `http://<instance-ip>:6006`

### Environment Variables

Configure optional services when launching the instance:

```bash
START_JUPYTER=true       # Enable Jupyter Lab on port 8888
START_TENSORBOARD=true   # Enable TensorBoard on port 6006
```

### Port Forwarding

VastAI automatically forwards ports, but you may need to configure firewall rules:

- **3000** - Next.js Frontend (Required)
- **8000** - FastAPI Backend (Required)
- **8888** - Jupyter Lab (Optional)
- **6006** - TensorBoard (Optional)

### Persistent Storage

VastAI instances can be configured with persistent storage volumes:

| Directory | Purpose | Recommended Size |
|-----------|---------|------------------|
| `/workspace/datasets` | Training images | 10-50 GB |
| `/workspace/output` | Trained LoRA files | 5-20 GB |
| `/workspace/pretrained_model` | Base models (SDXL, Flux, SD3) | 20-50 GB |
| `/workspace/vae` | VAE models | 1-5 GB |
| `/workspace/training_logs` | TensorBoard logs | 1-5 GB |

**Total Recommended**: 50-200 GB depending on usage

### Cost Optimization

1. **Use Interruptible Instances**: Save 50-70% for non-critical training
2. **Stop When Idle**: VastAI charges by the hour - stop instances when not training
3. **Download Models Once**: Keep base models in persistent storage
4. **Monitor GPU Utilization**: Ensure your batch size fully utilizes GPU

---

## 🐳 Local Docker Deployment

Run the trainer locally with Docker (requires NVIDIA GPU with CUDA support).

### Prerequisites

- Docker 20.10+ with NVIDIA Container Toolkit
- NVIDIA GPU with 12GB+ VRAM
- CUDA 12.1+ drivers
- 16GB+ system RAM
- 50GB+ free disk space

### Installation

1. **Install NVIDIA Container Toolkit**:
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. **Clone Repository**:
   ```bash
   git clone --recurse-submodules https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
   cd Ktiseos-Nyx-Trainer
   ```

3. **Build Docker Image**:
   ```bash
   docker build -t ktiseos-nyx-trainer .
   ```

4. **Run Container**:
   ```bash
   docker run -d \
     --name ktiseos-trainer \
     --gpus all \
     -p 3000:3000 \
     -p 8000:8000 \
     -p 8888:8888 \
     -p 6006:6006 \
     -v $(pwd)/datasets:/workspace/datasets \
     -v $(pwd)/output:/workspace/output \
     -v $(pwd)/pretrained_model:/workspace/pretrained_model \
     -v $(pwd)/vae:/workspace/vae \
     -v $(pwd)/training_logs:/workspace/training_logs \
     -e START_JUPYTER=true \
     -e START_TENSORBOARD=true \
     ktiseos-nyx-trainer
   ```

5. **Access Services**:
   - Web UI: http://localhost:3000
   - API: http://localhost:8000
   - Jupyter: http://localhost:8888
   - TensorBoard: http://localhost:6006

### Docker Compose (Easier Management)

Use the included `docker-compose.yml`:

```bash
docker-compose up -d
```

To stop:
```bash
docker-compose down
```

### Monitoring Logs

```bash
# View all logs
docker logs -f ktiseos-trainer

# View specific service logs
docker exec ktiseos-trainer tail -f /workspace/logs/backend.log
docker exec ktiseos-trainer tail -f /workspace/logs/frontend.log
```

---

## 💻 Development Setup (No Docker)

For local development without Docker.

### Prerequisites

- Python 3.11
- Node.js 20.x
- NVIDIA GPU with CUDA 12.1+ (for training)
- Git with LFS support

### Backend Setup

1. **Clone with Submodules**:
   ```bash
   git clone --recurse-submodules https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
   cd Ktiseos-Nyx-Trainer
   ```

2. **Run Installer**:
   ```bash
   python installer.py
   ```

3. **Install API Dependencies**:
   ```bash
   pip install -r requirements-api.txt
   ```

4. **Start FastAPI Backend**:
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Frontend Setup

1. **Install Dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Run Development Server**:
   ```bash
   npm run dev
   ```

3. **Build for Production**:
   ```bash
   npm run build
   npm run start
   ```

### Development URLs

- Frontend Dev: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## 🔧 Configuration

### Backend Configuration

Edit `api/main.py` to configure:
- CORS origins
- Upload limits
- Logging levels

### Frontend Configuration

Edit `frontend/next.config.js` for:
- API endpoint URLs
- Build optimization
- Static export settings

### Training Configuration

Default paths can be set in environment variables:

```bash
export WORKSPACE_DIR=/workspace
export DATASETS_DIR=/workspace/datasets
export OUTPUT_DIR=/workspace/output
export PRETRAINED_MODEL_DIR=/workspace/pretrained_model
```

---

## 🐛 Troubleshooting

### GPU Not Detected

**Symptom**: Training fails or uses CPU

**Solution**:
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA availability in container
docker exec ktiseos-trainer python -c "import torch; print(torch.cuda.is_available())"

# Ensure --gpus all flag is used
docker run --gpus all ...
```

### Port Already in Use

**Symptom**: `Error: bind: address already in use`

**Solution**:
```bash
# Find process using port
sudo lsof -i :3000
sudo lsof -i :8000

# Kill process or use different port
docker run -p 3001:3000 -p 8001:8000 ...
```

### Out of Memory During Training

**Symptom**: CUDA OOM errors

**Solution**:
- Reduce `train_batch_size` in training config
- Enable `gradient_checkpointing`
- Use `cache_latents` and `cache_latents_to_disk`
- Enable FP8 training for memory savings

### Submodule Issues

**Symptom**: `sd_scripts` directory missing

**Solution**:
```bash
git submodule update --init --recursive
```

### Frontend Build Failures

**Symptom**: npm build errors

**Solution**:
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install

# Clear Next.js cache
rm -rf .next
npm run build
```

---

## 📊 Performance Tuning

### SDXL Training Benchmarks

| GPU | VRAM | Batch Size | Gradient Checkpointing | Time per Epoch (1000 images) |
|-----|------|------------|------------------------|------------------------------|
| RTX 3060 12GB | 12GB | 1 | Yes | ~45 min |
| RTX 4060 Ti 16GB | 16GB | 2 | Yes | ~30 min |
| RTX 3090 24GB | 24GB | 4 | No | ~15 min |
| RTX 4090 24GB | 24GB | 4 | No | ~12 min |
| A100 40GB | 40GB | 8 | No | ~8 min |

### Optimal Settings by GPU

**12GB VRAM (RTX 3060, RTX 3060 Ti)**:
- Batch Size: 1-2
- Gradient Checkpointing: Enabled
- Mixed Precision: fp16
- Cache Latents: Enabled
- Resolution: 1024 (SDXL) or 512 (SD 1.5)

**16-24GB VRAM (RTX 4060 Ti, RTX 3090, RTX 4090)**:
- Batch Size: 2-4
- Gradient Checkpointing: Optional
- Mixed Precision: fp16
- Cache Latents: Enabled
- Resolution: 1024 (SDXL) or 768 (SD 1.5)

**40GB+ VRAM (A100, A6000)**:
- Batch Size: 4-8
- Gradient Checkpointing: Disabled
- Mixed Precision: fp16 or bf16
- Cache Latents: Optional
- Resolution: 1024+ (SDXL)

---

## 🔐 Security Considerations

### Production Deployment

If deploying publicly, add authentication:

1. **Backend**: Add FastAPI authentication middleware
2. **Frontend**: Add NextAuth.js or similar
3. **Firewall**: Restrict ports 8000, 8888, 6006 to trusted IPs
4. **HTTPS**: Use reverse proxy (nginx, Caddy) with SSL

### VastAI Security

- VastAI instances are publicly accessible by default
- Use strong passwords for Jupyter if enabled
- Don't store sensitive data in containers
- Download trained models and delete instances when done

---

## 📚 Additional Resources

- [VastAI Documentation](https://docs.vast.ai)
- [Kohya sd-scripts Guide](https://github.com/kohya-ss/sd-scripts)
- [LyCORIS Documentation](https://github.com/KohakuBlueleaf/LyCORIS)
- [SDXL LoRA Training Guide](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/blob/main/docs/TRAINING_GUIDE.md)

---

## 🆘 Support

- **GitHub Issues**: [Report bugs and feature requests](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues)
- **Discussions**: [Community Q&A](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/discussions)
