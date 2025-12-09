# 🗺️ Ktiseos Nyx Trainer - Product Roadmap

**Last Updated:** 2025-11-30
**Current Phase:** Alpha - Core Features & Polish
**Next Release:** v0.2.0 (Multi-Trainer Support)

---

## 🎯 Vision

Build the **most accessible and powerful LoRA training platform** by combining:
- Modern web UI (Next.js + Tailwind) - no Gradio!
- Modular backend services (FastAPI + Pydantic)
- Multiple trainer backends (Kohya, official variants, future extensions)
- Neurodivergent-friendly UX with clear workflows
- VastAI/Runpod optimized for cloud training

---

## 📊 Current Status (v0.1.0)

### ✅ **Completed**
- [x] Modern Next.js frontend with Tailwind
- [x] FastAPI backend with service layer architecture
- [x] Vendored Derrian backend (Kohya + LyCORIS + custom optimizers)
- [x] Dataset upload, tagging (WD14), caption management
- [x] Training configuration with all SDXL/SD1.5 parameters exposed
- [x] Real-time training monitoring via WebSocket
- [x] Model/VAE downloads from HuggingFace/Civitai
- [x] LoRA utilities (resize, upload to HF)
- [x] Docker deployment for VastAI/Runpod
- [x] Tag editor with image display
- [x] Download progress indicators (basic)

### 🚧 **In Progress**
- [ ] Full Flux parameter support
- [ ] Async downloads with real progress tracking
- [ ] UI polish and accessibility improvements

### 🐛 **Known Issues**
- Downloads block backend (no async/cancellation)
- Some Flux-specific params missing from UI
- Need better error handling in training workflow

---

## 🎯 Phases

### **Phase 1: Alpha - Core Features** ✅ (Current)
**Goal:** Stable core training workflow for SDXL/SD1.5

**Status:** 95% Complete
- ✅ Basic training pipeline works
- ✅ All major features implemented
- 🚧 UI polish ongoing
- 🚧 Bug fixes and stability

**Remaining:**
- [ ] Test full workflow on VastAI
- [ ] Document deployment process
- [ ] Fix remaining UI bugs

---

### **Phase 2: Multi-Trainer Support** 🎯 (Next - v0.2.0)
**Goal:** Support multiple training backends via strategy pattern

**Timeline:** 2-3 weeks
**Priority:** High

**Features:**
1. **Architecture:**
   - [x] BaseTrainer pattern (already exists!)
   - [ ] Trainer selection in UI
   - [ ] Config translation layer
   - [ ] Backend isolation (separate venvs if needed)

2. **Trainers to Add:**
   - [ ] **Pure Kohya** (official sd-scripts, no Derrian mods)
   - [ ] **Kohya + Custom Optimizers** (current Derrian setup)
   - [ ] **Future:** ComfyUI training nodes
   - [ ] **Future:** Alternative frameworks

3. **Implementation Plan:**
   - Vendor each trainer backend separately (`trainer/kohya_official/`, etc.)
   - Create trainer classes implementing BaseTrainer
   - Add UI dropdown for trainer selection
   - Map frontend config → trainer-specific TOML/args

**See:** `docs/planning/architecture/MULTI_TRAINER_ARCHITECTURE.md`

---

### **Phase 3: Flux & Advanced Model Support** (v0.3.0)
**Goal:** Full Flux, SD3, and future model support

**Timeline:** 3-4 weeks
**Priority:** Medium-High

**Features:**
1. **Flux Training:**
   - [ ] Add missing Flux parameters (discrete_flow_shift, timestep_sampling, etc.)
   - [ ] Block-wise training for Flux (double/single blocks)
   - [ ] T5-XXL configuration
   - [ ] Flow-based sampling modes

2. **SD3 Support:**
   - [ ] SD3-specific parameters
   - [ ] Triple text encoder support
   - [ ] MMDiT architecture params

3. **Model Management:**
   - [ ] Better model version detection
   - [ ] Automatic parameter adjustment based on model type
   - [ ] Model compatibility checking

**See:** `docs/planning/features/TODO_KOHYA_GUI_PATTERNS.md` (Flux section)

---

### **Phase 4: UX & Performance** (v0.4.0)
**Goal:** Polish UX and improve performance

**Timeline:** 2-3 weeks
**Priority:** Medium

**Features:**
1. **Async Downloads:**
   - [ ] Background download jobs with job tracking
   - [ ] Real-time progress bars (%, speed, ETA)
   - [ ] Download cancellation
   - [ ] Multiple simultaneous downloads
   - [ ] Resume failed downloads

2. **UI Polish:**
   - [ ] Improved parameter organization
   - [ ] Contextual help tooltips
   - [ ] Keyboard shortcuts
   - [ ] Better mobile responsiveness
   - [ ] Dark/light mode toggle

3. **Performance:**
   - [ ] Frontend optimization (code splitting, lazy loading)
   - [ ] Backend caching
   - [ ] Database for job history (SQLite)
   - [ ] Training templates/presets

**See:**
- `docs/planning/features/TODO_ASYNC_DOWNLOADS.md`
- `docs/planning/features/TODO_UI_POLISH.md`

---

### **Phase 5: Advanced Features** (v0.5.0)
**Goal:** Power user features and extensibility

**Timeline:** 4-6 weeks
**Priority:** Medium-Low

**Features:**
1. **Captioning Alternatives:**
   - [ ] BLIP/BLIP2 integration (photography captions)
   - [ ] Custom caption models
   - [ ] Batch caption operations
   - [ ] Caption quality scoring

2. **Dataset Tools:**
   - [ ] Image preprocessing (resize, crop, color correction)
   - [ ] Duplicate detection
   - [ ] Quality filtering
   - [ ] Auto-tagging confidence thresholds
   - [ ] Tag hierarchies and aliases

3. **Training Enhancements:**
   - [ ] Training templates library
   - [ ] Parameter search/optimization
   - [ ] Multi-stage training workflows
   - [ ] Automated quality checking

4. **Inference Integration:**
   - [ ] Quick test inference during/after training
   - [ ] Side-by-side comparison
   - [ ] A/B testing interface

**See:** `docs/planning/features/ADVANCED_FEATURES.md` (to be created)

---

## 🔧 Technical Debt & Refactoring

### **High Priority:**
- [ ] Add comprehensive error handling across services
- [ ] Implement proper logging system
- [ ] Add unit tests for core services
- [ ] API documentation (OpenAPI/Swagger)

### **Medium Priority:**
- [ ] Frontend state management (consider Zustand)
- [ ] API client generation from OpenAPI spec
- [ ] Better TypeScript types for training configs
- [ ] Component library documentation

### **Low Priority:**
- [ ] E2E testing with Playwright
- [ ] Performance monitoring
- [ ] Analytics integration
- [ ] Internationalization (i18n)

---

## 📦 Release Schedule

### **v0.1.0** - Alpha Release ✅
- **Date:** 2025-11-30
- **Status:** Complete
- **Focus:** Core training workflow

### **v0.2.0** - Multi-Trainer Support
- **Target:** ~2025-12-15
- **Status:** Planning
- **Focus:** Multiple training backends

### **v0.3.0** - Flux & Advanced Models
- **Target:** ~2026-01-15
- **Status:** Planned
- **Focus:** Full Flux support, SD3

### **v0.4.0** - UX & Performance
- **Target:** ~2026-02-01
- **Status:** Planned
- **Focus:** Async downloads, UI polish

### **v0.5.0** - Advanced Features
- **Target:** ~2026-03-15
- **Status:** Planned
- **Focus:** BLIP captioning, advanced dataset tools

### **v1.0.0** - Stable Release
- **Target:** ~2026-04-01
- **Status:** Vision
- **Focus:** Production-ready, fully tested, documented

---

## 🎯 Success Metrics

### **Phase 1 (Alpha):**
- [x] Can complete full training workflow locally
- [ ] Can deploy and train on VastAI
- [ ] Zero critical bugs in core workflow
- [ ] Documentation covers all features

### **Phase 2 (Multi-Trainer):**
- [ ] Support 2+ trainer backends
- [ ] Users can switch trainers via UI
- [ ] Each trainer produces valid LoRAs

### **Phase 3 (Flux):**
- [ ] Full Flux parameter coverage
- [ ] Successful Flux LoRA training
- [ ] SD3 basic support

### **Phase 4 (UX):**
- [ ] All operations have progress feedback
- [ ] UI is polished and intuitive
- [ ] Performance meets targets (page load <2s)

### **Phase 5 (Advanced):**
- [ ] BLIP captioning integrated
- [ ] Advanced dataset tools working
- [ ] Power users report high satisfaction

### **v1.0:**
- [ ] 100+ successful trainings completed
- [ ] Community adoption (GitHub stars, forks)
- [ ] Production deployments
- [ ] Comprehensive documentation

---

## 🤝 Contributing

Want to help? Check:
- `docs/planning/features/` - Feature-specific plans
- `docs/planning/architecture/` - Architecture decisions
- `CLAUDE.md` - Development guidelines

---

## 📚 Documentation Structure

```
docs/
├── planning/
│   ├── features/           # Feature-specific TODOs
│   ├── architecture/       # Architecture decisions
│   └── completed/          # Completed refactors/plans
├── api/                    # API documentation
├── deployment/             # Deployment guides
└── user-guides/            # User documentation
```

---

## 🔗 Quick Links

- **Current Priorities:** See sections marked with 🎯
- **Next Steps:** Phase 2 - Multi-Trainer Support
- **Feature Requests:** Open an issue on GitHub
- **Architecture Decisions:** See `docs/planning/architecture/`

---

**Last Review:** 2025-11-30
**Next Review:** 2025-12-07 (weekly reviews during alpha)
