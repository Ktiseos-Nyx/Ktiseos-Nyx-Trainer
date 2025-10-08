# Project Vision: A Pluggable AI Training Ecosystem

The primary long-term goal is to evolve this project from a single-backend tool into a flexible, multi-backend ecosystem. The architecture will be refactored to support different training "engines" (like `kohya-ss`, `Simple Tuner`, etc.) under a unified, user-friendly interface.

### Core Principles:
- **Abstracted UI:** The user experience in the notebooks will be consistent, regardless of the backend selected.
- **Pluggable Backends:** Each training engine will be a self-contained component with its own installer and manager logic.
- **Simplicity and Accessibility:** Maintain the project's focus on providing a less overwhelming, more guided experience than traditional GUIs.

---

# 🗺️ LoRA Easy Training - Development Roadmap

This roadmap outlines planned improvements and features for the LoRA Easy Training system. Items are organized by priority and development complexity.

## 🎯 High Priority - User-Facing Features

### 1. Checkpoint Finetuning Support
**Status:** Planned  
**Impact:** High - Opens entirely new training paradigm

Add support for full model finetuning (modifying base model weights) in addition to current LoRA adapter training.

- **Implementation:** Leverage existing Kohya scripts (`sdxl_train.py`, `flux_train.py`, etc.)
- **New Features:** Different parameter widgets, memory profile adjustments, larger output handling
- **User Benefits:** Create custom base models instead of just adaptations
- **UI/UX:** Create a new, dedicated `Finetuning_Trainer.ipynb` to provide a focused workflow for this advanced task, keeping the main LoRA notebook simple.
- **Challenges:** Higher VRAM requirements, longer training times, larger output files

### 2. LoRA Training Profiles  
**Status:** Research Phase  
**Impact:** High - Significantly improves user success rate

Curated, battle-tested training configurations based on community knowledge.

- **Research Sources:** Bmaltais Kohya GUI, Civitai community practices, successful LoRA creators
- **Profile Categories:** Character training, style training, concept learning, photorealism, anime/illustration
- **Implementation:** Pre-configured parameter sets with explanations and use case guidance
- **User Benefits:** Eliminate guesswork, proven configurations, faster setup

## 🔧 Medium Priority - Code Quality & UX

### 3. Training Manager Refactor
**Status:** Planned  
**Impact:** Medium - Developer experience and maintainability

Break down the 1500-line training manager into focused, maintainable components.

- **Proposed Structure:** 
  - `ModelDetector` - Model type detection and validation
  - `ConfigGenerator` - TOML generation and parameter mapping
  - `TrainingExecutor` - Process execution and monitoring
  - `ProfileManager` - Memory and training profile management
  - `OptimizerRegistry` - Optimizer and LyCORIS method configurations
- **Benefits:** Easier testing, clearer responsibilities, better code reuse

### 4. CSS Theming System
**Status:** Planned  
**Impact:** Medium - Professional appearance and consistency

Custom CSS themes for notebook widgets instead of relying on inconsistent Jupyter themes.

- **Theme Options:** Dark mode, light mode, minimal, cyberpunk
- **Features:** Consistent styling across environments, theme selector widget, automatic detection
- **Benefits:** Professional appearance, user choice, easier maintenance

### 5. Modularize Utility Notebooks
**Status:** Planned
**Impact:** Medium - Improves usability and organization

Break down the monolithic `Utilities_Notebook.ipynb` into smaller, single-purpose notebooks, each with a more focused and spacious UI.

- **Proposed Split:**
  - `Checkpoint_Merger.ipynb`: A dedicated tool for merging full model checkpoints.
  - `Lora_Utilities.ipynb`: For LoRA-specific tasks like merging weights, extracting LoRAs, and resizing.
  - `Model_Converter.ipynb`: For converting model formats (e.g., `.safetensors` to `.ckpt`).
  - `File_Uploader.ipynb`: A standalone, large-interface notebook for general-purpose file uploads and management, moving it out of other notebooks.
- **User Benefits:** Easier to find the specific tool needed; less overwhelming than a single large notebook with many functions.

## 🔍 Research & Improvement

### 6. Kohya/LyCORIS Settings Audit
**Status:** Ongoing  
**Impact:** Medium - Feature completeness and training quality

Comprehensive review of available settings to ensure feature parity with latest Kohya developments.

- **Research Areas:** New training arguments, advanced noise settings, memory optimizations, scheduler parameters
- **LyCORIS:** Algorithm-specific parameters, new methods, better defaults
- **Validation:** Improved parameter checking, better error messages, user guidance

### 7. Optimizer & Memory Efficiency
**Status:** Verification Needed  
**Impact:** Medium - Training accessibility and reliability

Ensure 8-bit optimizers and memory optimizations work correctly across environments.

- **Bits and Bytes:** Verify installation, CUDA compatibility, actual memory savings
- **Testing:** Container environments (VastAI, RunPod), different hardware configurations
- **Fallbacks:** Graceful degradation when optimizations unavailable

## 🌐 Platform Compatibility

### 8. ROCm Support (AMD GPU)
**Status:** Planned  
**Impact:** High - Expands user base significantly

Add support for AMD GPUs using ROCm instead of CUDA.

- **Implementation:** ROCm-specific installation paths, environment detection, memory management
- **Challenges:** Different APIs, optimizer compatibility, performance differences
- **Benefits:** Open training to AMD GPU users currently unable to use the system

### 9. Custom Docker Container
**Status:** Seeking Community Contributor
**Impact:** High - Massively improves stability and ease of use.

Create and maintain a custom Docker container specifically for the Ktiseos-Nyx Trainer ecosystem.

- **Goal:** To provide a single, pre-configured environment with all necessary dependencies, CUDA versions, and Python packages locked and tested.
- **User Benefits:**
  - Eliminates environment and dependency-related errors ("dependency hell").
  - Guarantees a stable, reproducible training environment.
  - Simplifies setup to a single `docker run` command.
- **Community Call:** We are actively seeking a contributor with Docker expertise to help lead this effort.

### 10. Intel Arc Research
**Status:** Research Phase  
**Impact:** Low - Experimental platform support

Investigate Intel Arc GPU support options.

- **Option A:** Wait for Kohya to support Intel XPU (unlikely)
- **Option B:** Parallel HuggingFace-based training system (significant development)
- **Assessment:** Small user base, experimental drivers, significant implementation complexity
- **Priority:** Research only, implementation depends on user demand

## 🛠️ Technical Improvements

### Error Handling Enhancement
**Status:** Under Review  
Implement structured exception handling with custom error classes for better debugging and user feedback.

### Model Detection Improvements  
**Status:** Under Review  
Enhanced model type detection using file headers instead of relying solely on filename patterns.

### Memory Profile Optimization
**Status:** Low Priority  
Current 4-profile system works well; expansion would add complexity without clear user benefit.

---

## 📅 Development Phases

### Phase 1: Core Features (Q1-Q2)
- Checkpoint finetuning support
- LoRA training profiles research and implementation
- ROCm support foundation

### Phase 2: Quality of Life (Q2-Q3)  
- CSS theming system
- Settings audit and improvements
- Enhanced error handling

### Phase 3: Architecture (Q3-Q4)
- Training manager refactor
- Platform compatibility expansion
- Advanced feature research

---

## 🤝 Community Input

We welcome community feedback on these roadmap items:

- **Feature Requests:** What training capabilities are you missing?
- **Platform Needs:** Which GPU platforms should we prioritize?
- **Training Profiles:** Share your successful training configurations
- **User Experience:** What parts of the current system are confusing or error-prone?

Join our [Discord](https://discord.gg/HhBSM9gBY) or open GitHub discussions to contribute ideas and feedback.

---

*Last Updated: January 2025*  
*This roadmap represents current planning and may change based on community needs, technical constraints, and development resources.*