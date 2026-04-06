# Ktiseos-Nyx LoRA Training Backend

## Overview

This is a **fork** of the original [Derrian's LoRA Easy Training Scripts Backend](https://github.com/derrian-distro/LoRA_Easy_Training_scripts_Backend), specifically adapted for **server and Jupyter notebook environments**.

## Purpose

While the original backend was designed primarily for **Google Colab** environments, this fork is optimized for:

- 🖥️ **Server deployments** (VastAI, RunPod, dedicated servers)
- 📚 **Jupyter notebook integration** 
- 🔄 **Multi-session workflows** with persistent configurations
- 🛠️ **Professional training environments** with proper path handling

## Key Differences from Original

### Path Handling
- ✅ **Absolute path resolution** instead of relative path assumptions
- ✅ **Working directory independence** - scripts work regardless of call location  
- ✅ **Flexible project structure** support

### Environment Compatibility
- 🎯 **Jupyter-first design** - built for notebook workflows
- 🐳 **Container-friendly** - works in Docker/containerized environments
- 💾 **Persistent storage** assumptions instead of ephemeral Colab storage

### Enhanced Features
- 🔧 **Improved error handling** for server environments
- 📊 **Extended LoRA variant support** (DoRA, DyLoRA, LoHa, etc.)
- 🚀 **Optimized for repeated training sessions**

## Compatibility

### ✅ Designed For
- **Ktiseos-Nyx LoRA Easy Training Jupyter** notebooks
- Server-based training environments
- Jupyter Lab/Notebook environments
- Containerized deployments (Docker, VastAI, RunPod)

### ❓ Theoretical Compatibility
This backend *could theoretically* work with **Derrian's original front-end**, but:
- ⚠️ **No guarantees** - different path assumptions may cause issues
- ⚠️ **Different target environments** - optimizations may not align
- ⚠️ **Untested combination** - use at your own risk

## Credits

- **Original Backend**: [Derrian](https://github.com/derrian-distro) - Created the foundational training backend
- **Fork Maintainer**: [Ktiseos-Nyx](https://github.com/Ktiseos-Nyx) - Server/Jupyter adaptations and enhancements
- **Core Training Scripts**: Based on [Kohya's sd-scripts](https://github.com/kohya-ss/sd-scripts)
- **LyCORIS Integration**: [KohakuBlueleaf's LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS)

## Installation

This backend is designed to be used as a **git submodule** within the main Jupyter training environment. It should not typically be installed standalone.

For the complete training environment, see: [LoRA Easy Training Jupyter](https://github.com/Ktiseos-Nyx/Lora_Easy_Training_Jupyter)

## Contributing

This is a specialized fork for server/Jupyter environments. For the original Colab-focused backend, please contribute to [Derrian's original repository](https://github.com/derrian-distro/LoRA_Easy_Training_scripts_Backend).

## License

Inherits the license from the original backend. See the original repository for license details.

---

*This fork exists to serve different use cases than the original. Both versions are valuable for their respective target environments.*
