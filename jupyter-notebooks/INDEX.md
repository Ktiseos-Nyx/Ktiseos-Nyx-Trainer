# Jupyter Notebooks - Quick Index

> 📓 Classic widget-based workflow for LoRA training

## 📚 Notebooks

| Notebook | Purpose | When to Use |
|----------|---------|-------------|
| **[Unified_LoRA_Trainer.ipynb](Unified_LoRA_Trainer.ipynb)** | Complete training pipeline | First time users, all-in-one workflow |
| **[Dataset_Preparation.ipynb](Dataset_Preparation.ipynb)** | Dataset management only | Preparing datasets separately |
| **[Utilities_Notebook.ipynb](Utilities_Notebook.ipynb)** | Post-training tools | After training is complete |

## 📖 Documentation

| Guide | Content |
|-------|---------|
| **[README.md](README.md)** | Complete Jupyter workflow guide |
| **[QUICKSTART.md](QUICKSTART.md)** | 5-minute setup for local users |
| **[VASTAI_JUPYTER.md](VASTAI_JUPYTER.md)** | VastAI-specific instructions |

## 🚀 Quick Start

### Local Users
```bash
python installer.py
jupyter lab
# Open Unified_LoRA_Trainer.ipynb
```

### VastAI Users
```
# Jupyter already running!
# Just open: http://<instance-ip>:8888
# Navigate to: Ktiseos-Nyx-Trainer/jupyter-notebooks/
```

## 🆚 Jupyter vs Web UI

| Feature | Jupyter Notebooks | Web UI |
|---------|-------------------|--------|
| **Interface** | Widget-based, cell execution | Modern web app |
| **Learning Curve** | Familiar to data scientists | Easier for beginners |
| **Customization** | Modify code inline | Settings page |
| **Deployment** | Local or cloud Jupyter | VastAI, Docker, local server |
| **File Management** | Basic upload widget | Full file browser |
| **Monitoring** | Progress widgets | Real-time WebSocket logs |
| **Best For** | Experimentation, step-by-step | Production, teams, servers |

**Both share the same backend** - use whichever you prefer!

## 💡 Pro Tips

1. **Web UI exists**: If you prefer modern interfaces, check out `../frontend/`
2. **Shared data**: Both workflows access same datasets/models
3. **Switch anytime**: Prepare dataset in Jupyter, train in Web UI (or vice versa)
4. **VastAI users**: Both Jupyter and Web UI work on the same instance

## 🔗 Main Project

- **Main README**: [../README.md](../README.md)
- **Web UI Guide**: [../docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md)
- **VastAI Setup**: [../VASTAI_SETUP.md](../VASTAI_SETUP.md)
- **GitHub**: https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer

---

**Choose your workflow and start training!** 🎉
