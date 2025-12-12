# Ktiseos-Nyx-Trainer - NextJS Training Environment

> **[► Create an Instance](https://cloud.vast.ai/?ref_id=70354&creator_id=70354&name=Ktiseos-Nyx-Trainer)**

## What is this template?

This template gives you a **web-based training environment** for Stable Diffusion, SDXL, Flux, and SD3.5 models. It comes with a Next.js frontend, FastAPI backend, and training infrastructure pre-configured - upload your images and start training.

**Think:** *"Model training and fine-tuning in your browser, powered by Kohya sd-scripts."*

> **Note:** This template uses a vendored backend approach (no git submodules) for maximum stability across cloud environments. Everything you need is included and ready to run.

---

## What can I do with this?

- **Train and fine-tune models** - LoRA, LyCORIS, and other adapters for SD1.5, SDXL, Flux, and SD3.5
- **Web interface** with Next.js frontend and FastAPI backend
- **Dataset preparation** with WD14 auto-tagging and caption management
- **Real-time monitoring** with TensorBoard integration
- **Jupyter environment** for file management, debugging, and custom workflows
- **File synchronization** across devices with Syncthing
- **Supervisor-managed services** that start on boot and restart on failure

---

## Who is this for?

This works well if you:
- Want to train custom models without complex setup
- Need a web-based training interface
- Are training characters, styles, concepts, or other adapters for image generation
- Want Kohya training scripts with a web UI
- Need a stable training environment on VastAI

---

## Quick Start Guide

### **Step 1: Launch Your Instance**
Click the **[Rent](https://cloud.vast.ai/?ref_id=YOUR_REF_ID&creator_id=YOUR_CREATOR_ID&name=Ktiseos-Nyx-Trainer)** button after selecting your GPU

### **Step 2: Wait for Provisioning**
The first boot takes **5-10 minutes** while the provisioning script:
- Clones the repository from GitHub
- Installs Python dependencies and Node.js 20+
- Builds the Next.js frontend
- Configures supervisor services
- Sets up the training environment

> **💡 Tip:** Watch the instance logs to monitor provisioning progress

### **Step 3: Access Your Services**
Once provisioning completes, the VastAI portal automatically creates access buttons for each service. Click these buttons in your instance dashboard:
- **"Frontend"** - Next.js training interface (port 3000)
- **"API"** - FastAPI backend documentation (port 8000)
- **"Jupyter"** - File management and development environment (port 8080)
- **"TensorBoard"** - Training metrics visualization (port 6006)

> **How it works:** Each button opens a secure Cloudflare tunnel to the corresponding service on your instance. No manual port forwarding or SSH tunneling required!

### **Step 4: Start Training!**
1. Upload your dataset images via the frontend
2. Run WD14 auto-tagging to generate captions
3. Configure training parameters (learning rate, epochs, etc.)
4. Start training and monitor progress in TensorBoard
5. Download your trained model when complete!

---

## Key Features

### **Web Interface**
- **Next.js Frontend** provides a responsive UI
- **FastAPI Backend** handles training operations and dataset management
- **Real-time updates** for training progress and status
- **Mobile-friendly** design works on tablets and phones

### **Dataset Preparation Tools**
- **WD14 Auto-Tagging** with ONNX runtime optimization
- **Caption Editor** for fine-tuning image descriptions
- **Folder Organization** with automatic directory management
- **Gallery-DL Integration** for downloading training images
- **HuggingFace Upload** for sharing datasets

### **Training Infrastructure**
- **Kohya sd-scripts** (vendored) for model training and fine-tuning
- **LyCORIS Integration** for LoRA, LoCon, LoHa, and other adapter architectures
- **TOML Configuration** system for reproducible training
- **Multi-architecture Support** - SD1.5, SDXL, Flux, SD3.5
- **Automatic checkpointing** and training resume

### **Experiment Tracking**
- **TensorBoard** integration for real-time metrics
- **Training Logs** accessible via frontend and Jupyter
- **Sample Generation** during training (coming soon)
- **Model Comparison** tools for evaluating results

### **Development Tools**
- **Jupyter** for file management and debugging
- **Python Virtual Environment** with all dependencies
- **SSH Access** for terminal-based workflows
- **Git Integration** for version control

### **Service Management**
- **Supervisor** manages all background services automatically
- Services start on boot and restart on failure
- Check status: `supervisorctl status ktiseos-nyx`
- View logs: `tail -f /var/log/portal/ktiseos-nyx.log`

### **VastAI Portal Integration**
- **Automatic service discovery** - Portal buttons created from `PORTAL_CONFIG`
- **Cloudflare tunnels** - Secure HTTPS access to all services without manual setup
- **Instance logs** - Monitor provisioning, startup, and errors from the VastAI dashboard
- **One-click access** - Each portal button routes to the correct port automatically
- **No port forwarding needed** - VastAI handles all networking through their tunnel system

### **Dynamic Provisioning**
The template uses a **provisioning script** that runs on first boot:
- Clones latest code from GitHub (`main` branch)
- Installs all Python and Node.js dependencies
- Builds the frontend for production
- Configures supervisor services
- Sets up the instance portal

**Provisioning Script URL:**
```
https://raw.githubusercontent.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/main/vastai_setup.sh
```

---

## Services & Ports

The VastAI portal system automatically creates access buttons for each service based on the `PORTAL_CONFIG` environment variable. Each button opens a tunnel to the corresponding port on your instance.

| Service | Port | Access Button | Description |
|---------|------|---------------|-------------|
| **Instance Portal** | 1111 | "Open" | VastAI application manager |
| **Next.js Frontend** | 3000 | "Frontend" | Main training interface |
| **FastAPI Backend** | 8000 | "API" | REST API and documentation |
| **Jupyter** | 8080 | "Jupyter" | File management and development environment |
| **TensorBoard** | 6006 | "TensorBoard" | Training metrics visualization |
| **Syncthing** | 8384 | "Syncthing" | File synchronization |

**How it works:** When you click a portal button (e.g., "Frontend"), VastAI routes you through their Cloudflare tunnel system to the corresponding port on your instance (e.g., port 3000). This provides secure HTTPS access without manual port forwarding or SSH tunneling.

**Instance Logs:** The VastAI portal also provides access to your instance logs. Click the "Logs" or "Console" option in your instance management page to monitor provisioning progress, service startup, and any errors.

---

## Directory Structure

After provisioning, your instance will have:

```
/workspace/Ktiseos-Nyx-Trainer/
├── frontend/              # Next.js web interface
├── api/                   # FastAPI backend
├── trainer/               # Training environment
│   └── derrian_backend/  # Vendored Kohya scripts + LyCORIS
├── core/                  # Core manager classes
├── pretrained_model/      # Downloaded base models
├── vae/                   # VAE files
├── training_config/       # Your training configurations (created on first use)
└── docs/                  # Documentation
```

---

## Customization Tips

### **Using Development Branch**
Want to test new features? Switch to the `dev` branch:
```bash
cd /workspace/Ktiseos-Nyx-Trainer
git checkout dev
git pull
supervisorctl restart ktiseos-nyx
```

### **Custom Provisioning**
Fork the repository and modify `vastai_setup.sh`, then update the environment variable:
```
PROVISIONING_SCRIPT=https://raw.githubusercontent.com/YOUR_USERNAME/Ktiseos-Nyx-Trainer/main/vastai_setup.sh
```

### **Environment Variables**
Configure your instance with these variables:
- `PROVISIONING_SCRIPT` - URL to setup script (required)
- `WORKSPACE` - Working directory (default: `/workspace`)
- `DATA_DIRECTORY` - Data storage location (default: `/workspace/`)

### **Adding Custom Dependencies**
```bash
# Activate the virtual environment
source /venv/main/bin/activate

# Install additional packages
pip install your-package-here

# Restart services if needed
supervisorctl restart ktiseos-nyx
```

### **Template Customization**
Want to save your configuration? Click **edit** in the VastAI template interface, make your changes, and save as your own template in **"My Templates"**.

---

## Troubleshooting

### **Services Not Starting?**
Check supervisor status:
```bash
supervisorctl status ktiseos-nyx
tail -f /var/log/portal/ktiseos-nyx.log
```

### **Frontend Build Failed?**
Ensure Node.js 20+ is installed:
```bash
node --version  # Should show v20.x.x or higher
cd /workspace/Ktiseos-Nyx-Trainer/frontend
npm run build
```

### **Training Errors?**
Check backend logs or use Jupyter for debugging:
```bash
# View backend logs
journalctl -u uvicorn -f

# Or access Jupyter for file inspection and debugging
# Click "Jupyter" button in Instance Portal
```

### **Need to Restart Everything?**
```bash
supervisorctl restart ktiseos-nyx
```

---

## Architecture Notes

### **Vendored Backend**
This project uses a **vendored backend** approach:
- All dependencies committed directly to the repository
- No git submodules (they caused issues on VastAI)
- Stable, predictable deployments
- Works reliably across Docker and bare-metal

### **Manager System**
Core functionality is organized into manager classes:
- `SetupManager` - Environment setup
- `ModelManager` - Model downloads
- `DatasetManager` - Dataset processing
- `KohyaTrainingManager` - Training execution
- `UtilitiesManager` - Post-training tools

---

## Need More Help?

- **Project Repository:** [GitHub - Ktiseos-Nyx-Trainer](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer)
- **Documentation:** [Project Docs](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/tree/main/docs)
- **Issues & Support:** [GitHub Issues](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues)
- **VastAI Support:** Use the messaging icon in the Vast.ai console
- **Instance Portal Guide:** [Vast.ai Instance Portal Documentation](https://docs.vast.ai/instance-portal)

---

**Template Version:** 1.0.0
**Last Updated:** 2025-12-12
**Maintained by:** Ktiseos-Nyx Team
