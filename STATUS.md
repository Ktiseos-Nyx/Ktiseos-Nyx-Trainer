# Development Status

Current status of Ktiseos Nyx LoRA Trainer development.

**Current Version:** Alpha (v0.1.0-dev)
**Last Updated:** 2026-03-03

---

## Overall Status: ALPHA

> This project is in active development. Features may not work as expected. Breaking changes may occur between updates.

## Feature Status

### Working Features

| Feature | Status | Notes |
|---------|--------|-------|
| **Web UI (Next.js)** | Working | Frontend functional |
| **Dataset Upload** | Working | Local and VastAI |
| **WD14 Auto-Tagging** | Working | Node.js ONNX (CPU) + Python (GPU) |
| **Caption Editor** | Working | Batch operations, trigger words |
| **File Browser** | Working | Tree view, file operations |
| **Training Config UI** | Working | 132+ parameters, 7 tabs, presets |
| **Config Persistence** | Working | RHF + localStorage (Zustand removed) |
| **TOML Generation** | Working | Both Node.js and Python paths |
| **Model Downloads** | Working | HuggingFace + Civitai |
| **Settings Management** | Working | HF token, Civitai key |
| **Windows Installation** | Working | `install.bat` |
| **Linux Installation** | Working | `installer.py` |
| **VastAI Deployment** | Working | Auto-setup via template |

### In Progress

| Feature | Status | Notes |
|---------|--------|-------|
| **Training Execution** | Stabilizing | Config generation works, debugging runtime edge cases |
| **Training Monitoring** | Stabilizing | Log polling works, event listener fixed |
| **Custom Optimizers (CAME)** | In Progress | PYTHONPATH fix applied, needs testing |
| **Validation Feedback** | Improved | Now shows field-level error details |

### Experimental

| Feature | Status | Notes |
|---------|--------|-------|
| **Flux.1 Training** | Experimental | Backend support exists, UI fields present |
| **SD3/SD3.5 Training** | Experimental | Minimal testing |
| **Lumina Training** | Experimental | Basic support |
| **BLIP/GIT Captioning** | Experimental | Python subprocesses |

### Planned

| Feature | Priority | Notes |
|---------|----------|-------|
| **Stable Flux Support** | Medium | Needs real-world testing |
| **Training Resume UI** | Low | Currently requires manual config |
| **Multi-LoRA Merging** | Low | Planned |
| **RunPod Support** | Low | Research |

## Platform Support

| Platform | Installation | Training | Status |
|----------|--------------|----------|--------|
| **Windows (NVIDIA)** | Yes | Yes | Supported |
| **Linux (NVIDIA)** | Yes | Yes | Supported |
| **VastAI** | Yes | Yes | Supported |
| **macOS** | Manual | No | UI only |
| **RunPod** | Untested | Untested | Experimental |
| **AMD ROCm** | No | No | Not implemented |

## How to Help

- [Report Issues](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues)
- [Discord Server](https://discord.gg/HhBSM9gBY)
- See [CONTRIBUTING.md](CONTRIBUTING.md)
