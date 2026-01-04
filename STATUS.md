# Development Status

Current status of Ktiseos Nyx LoRA Trainer development.

**Current Version:** Alpha (v0.1.0-dev)
**Last Updated:** 2026-01-03

---

## Overall Status: 🟡 ALPHA

> ⚠️ **This project is in active development.** Features may not work as expected. Breaking changes may occur between updates.

## Feature Status

### ✅ Stable Features

| Feature | Status | Notes |
|---------|--------|-------|
| **Web UI (Next.js)** | ✅ Stable | Frontend fully functional |
| **Backend API (FastAPI)** | ✅ Stable | REST API + WebSocket working |
| **Dataset Upload (Local)** | ✅ Stable | Works on Windows/Linux local |
| **WD14 Auto-Tagging** | ✅ Stable | All tagger models working |
| **Caption Editor** | ✅ Stable | Full editing functionality |
| **SD1.5 Training** | ✅ Stable | Well-tested, reliable |
| **SDXL Training** | ✅ Stable | Tested on 24GB VRAM |
| **Training Config UI** | ✅ Stable | 132 parameters, 7 tabs |
| **Real-time Monitoring** | ✅ Stable | WebSocket progress updates |
| **Model Downloads** | ✅ Stable | HuggingFace + Civitai |
| **LoRA Resizing** | ✅ Stable | SVD-based resizing |
| **Windows Installation** | ✅ Stable | `install.bat` works reliably |
| **Linux Installation** | ✅ Stable | `installer_local_linux.py` works |
| **VastAI Deployment** | ✅ Stable | Auto-setup via template |

### ⚠️ Known Issues

| Issue | Severity | Workaround | Tracking |
|-------|----------|------------|----------|
| **Dataset Upload on VastAI** | 🔴 High | Use VastAI Jupyter for uploads | [#77](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues/77) |
| **Frontend Build Size** | 🟡 Medium | Optimizations in progress | Addressed in #100 |
| **Windows Installer Version Detection** | 🟢 Low | Fixed in recent commit | Fixed |

### 🚧 Experimental Features

> These features are available in the Kohya backend but haven't been thoroughly tested in the UI.

| Feature | Status | Notes |
|---------|--------|-------|
| **Flux.1 Training** | ⚠️ Experimental | Backend support exists, UI untested |
| **SD3/SD3.5 Training** | ⚠️ Experimental | Requires sd3 branch, minimal UI testing |
| **Lumina Training** | ⚠️ Experimental | Basic support, needs validation |
| **Chroma Training** | ⚠️ Experimental | Minimal testing |
| **DoRA LoRA Type** | ⚠️ Experimental | Works but slower training |
| **BLIP/GIT Captioning** | ⚠️ Experimental | Alternative to WD14 tagging |

### 🔜 Planned Features

| Feature | Priority | Status | ETA |
|---------|----------|--------|-----|
| **Fix VastAI Upload** | 🔴 High | In Progress | Q1 2026 |
| **Frontend Size Optimization** | 🟡 Medium | In Progress | Q1 2026 |
| **Stable Flux Support** | 🟡 Medium | Research | Q2 2026 |
| **SD3.5 UI Integration** | 🟡 Medium | Planned | Q2 2026 |
| **Multi-LoRA Merging** | 🟢 Low | Planned | Q3 2026 |
| **Advanced Sampling** | 🟢 Low | Planned | Q3 2026 |
| **Training Resume UI** | 🟢 Low | Planned | TBD |
| **RunPod Support** | 🟢 Low | Research | TBD |
| **AMD ROCm Support** | 🟢 Low | Research | TBD |

## Platform Support Status

| Platform | Installation | Training | Status |
|----------|--------------|----------|--------|
| **Windows (NVIDIA)** | ✅ | ✅ | Fully supported |
| **Linux (NVIDIA)** | ✅ | ✅ | Fully supported |
| **macOS** | ⚠️ Manual | ❌ | UI only (no training) |
| **VastAI** | ✅ | ✅ | Fully supported (upload issues) |
| **RunPod** | ⚠️ Untested | ⚠️ Untested | Experimental |
| **AMD ROCm** | ❌ | ❌ | Not implemented |

## Recent Changes

### 2026-01-03
- ✅ Fixed Windows installer Python version detection
- ✅ Updated CLAUDE.md with Windows development environment
- ✅ Added frontend optimization (dynamic imports, standalone mode)
- ✅ Streamlined README and created detailed documentation

### 2025-12-27
- ✅ Fixed DoRA implementation in LyCORIS integration
- ✅ Added missing LyCORIS algorithms

### 2025-12-24
- ✅ Implemented proper onSave for training configs
- ✅ Fixed training config state management

### 2025-12-17
- ✅ Added development environment documentation
- ✅ Enabled Turbopack for faster frontend builds

## Known Limitations

### Technical
- **AMD GPUs**: Kohya SS (sd-scripts) only supports NVIDIA CUDA
- **Apple Silicon**: MPS (Metal) not supported by Kohya backend
- **VRAM**: SDXL requires 24GB, Flux requires 32GB+
- **Windows RAM**: Development builds need 16GB+ RAM

### Platform-Specific
- **VastAI**: Dataset uploader not working (use Jupyter workaround)
- **macOS**: Cannot train LoRAs (UI development only)
- **RunPod**: Not officially tested or supported

### Feature Gaps
- No multi-LoRA merging yet
- Training resume requires manual TOML editing
- No automatic hyperparameter tuning
- Limited sample generation during training

## Roadmap

### Q1 2026 (Jan-Mar)
- [ ] Fix VastAI dataset uploader
- [ ] Complete frontend optimization
- [ ] Stabilize Flux training
- [ ] Bug fixes and polish

### Q2 2026 (Apr-Jun)
- [ ] SD3.5 full support
- [ ] Advanced sampling UI
- [ ] Improved documentation
- [ ] Community feature requests

### Q3 2026 (Jul-Sep)
- [ ] Multi-LoRA merging
- [ ] Training resume from UI
- [ ] Dataset preprocessing pipeline
- [ ] Beta release candidate

### Q4 2026 (Oct-Dec)
- [ ] 1.0 Release
- [ ] Comprehensive testing
- [ ] Production hardening
- [ ] Performance optimization

## How to Help

**Reporting Issues:**
- Check [existing issues](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues) first
- Include error logs, system info, and steps to reproduce
- Use issue templates when available

**Contributing:**
- See [CONTRIBUTING.md](CONTRIBUTING.md)
- Join [Discord](https://discord.gg/HhBSM9gBY) for discussion
- Test experimental features and report results

**Feedback:**
- Feature requests on GitHub Issues
- Bug reports with detailed reproduction steps
- Documentation improvements via PR

## Support Status

| Component | Support Level |
|-----------|---------------|
| **Core Training** | ✅ Active development |
| **Web UI** | ✅ Active development |
| **Documentation** | ✅ Active updates |
| **VastAI Template** | ✅ Maintained |
| **RunPod** | ⚠️ Community-driven |
| **Experimental Features** | ⚠️ Best-effort |

---

**Stay Updated:**
- ⭐ Star the repo on [GitHub](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer)
- 📢 Join [Discord](https://discord.gg/HhBSM9gBY)
- 📝 Watch [CHANGELOG.md](CHANGELOG.md) for updates
