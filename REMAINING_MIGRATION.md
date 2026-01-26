# Remaining Python → Node.js Migration

**Last Updated**: 2026-01-26  
**Current Status**: 75% Complete (Phases 1-5 done)

## 📊 What's Left to Migrate

### ✅ Already Migrated (Node.js)
- File operations (`files.py` → Node.js API routes)
- Settings management (`settings.py` → Node.js API routes)
- Caption editing (`dataset.py` captions → Node.js API routes)
- Job management (`training.py` jobs → Node.js job-manager)
- WD14 Tagging (`dataset.py` tag → Node.js ONNX Runtime)

---

## 🔄 Can/Should Migrate to Node.js

### 1. Config Management Routes (Medium Priority)
**Current**: `api/routes/config.py` (10 routes)  
**Service**: Already have `config-service.ts` ✅  
**Missing**: API routes to expose it

**Routes to create:**
- `GET /api/config/templates` - List config templates
- `GET /api/config/load` - Load config from file
- `POST /api/config/save` - Save config to file
- `POST /api/config/save-training` - Save training config (generates TOML)
- `POST /api/config/validate` - Validate config
- `GET /api/config/defaults` - Get default config
- `GET /api/config/presets` - List presets
- `GET /api/config/presets/{id}` - Get specific preset
- `POST /api/config/presets` - Save preset
- `DELETE /api/config/presets/{id}` - Delete preset

**Complexity**: Low (service already exists, just need routes)  
**Benefit**: Completes config system migration

### 2. Model Downloads (Low Priority)
**Current**: `api/routes/models.py` (6 routes)  
**What it does**: Download models/VAEs from HuggingFace/Civitai

**Routes:**
- `POST /api/models/download` - Download model (spawns aria2c/wget)
- `GET /api/models/list` - List downloaded models
- `DELETE /api/models/{type}/{name}` - Delete model
- `POST /api/models/cancel` - Cancel download
- `GET /api/models/popular` - Get popular models list

**Migration approach:**
- Could use Node.js `child_process` to spawn aria2c/wget
- Or use Node.js libraries like `axios` with progress tracking
- Basically just file downloads + progress parsing

**Complexity**: Low-Medium (subprocess spawning + progress tracking)  
**Benefit**: Removes one more Python dependency

### 3. Civitai Integration (Low Priority)
**Current**: `api/routes/civitai.py` (5 routes)  
**What it does**: Browse Civitai, get model info, download

**Routes:**
- `GET /api/civitai/models` - Browse models
- `GET /api/civitai/models/{id}` - Get model details
- `GET /api/civitai/tags` - Get tags
- `GET /api/civitai/model-versions/{id}` - Get version info
- `POST /api/civitai/download` - Download model

**Migration approach:**
- All of this is just HTTP API calls to Civitai
- Can use `axios` or `fetch` in Node.js
- No ML dependencies

**Complexity**: Low (just HTTP calls)  
**Benefit**: Removes another Python file

### 4. Dataset Operations (Partial - Medium Priority)
**Current**: `api/routes/dataset.py` (26 routes - partially migrated)  
**Already in Node.js**: Caption editing, tag operations, image serving  
**Still in Python**: Dataset CRUD, uploads

**Routes to migrate:**
- `GET /api/dataset/list` - List datasets
- `POST /api/dataset/create` - Create dataset
- `GET /api/dataset/{name}` - Get dataset info
- `GET /api/dataset/{name}/files` - List dataset files
- `DELETE /api/dataset/{name}` - Delete dataset
- `POST /api/dataset/upload-batch` - Upload multiple files
- `POST /api/dataset/upload-zip` - Upload and extract ZIP
- `POST /api/dataset/download-url` - Download from URL

**Migration approach:**
- File uploads: Use Next.js API route with `formData`
- ZIP extraction: Use Node.js `unzipper` or `adm-zip`
- URL downloads: Use `axios` with streams

**Complexity**: Medium (file handling, multipart uploads)  
**Benefit**: Complete dataset management in Node.js

### 5. Utilities (Calculator, etc.) (Low Priority)
**Current**: `api/routes/utilities.py` (11 routes)  
**Mixed**: Some can migrate, some must stay Python

**Can migrate to Node.js:**
- `POST /api/utilities/calculator` - Step calculation (just math)
- `GET /api/utilities/datasets/browse` - Browse datasets (file ops)
- `GET /api/utilities/directories` - Get directories (file ops)
- `GET /api/utilities/lora/resize-dimensions` - Get dimensions (constants)
- `POST /api/utilities/lora/list` - List LoRA files (file ops)

**Complexity**: Low (no ML dependencies)

---

## ❌ Must Stay in Python (ML Operations)

These require PyTorch, transformers, or safetensors manipulation - **keep as Python subprocesses**.

### 1. Training (Keep as subprocess)
**File**: `api/routes/training.py` (5 routes)  
**Subprocess**: Kohya SS training scripts

**Routes:**
- `POST /api/training/start` - Spawn training subprocess ✅
- `POST /api/training/validate` - Validate config (could migrate)
- `GET /api/training/status/{id}` - Job status (already in job-manager)
- `POST /api/training/stop/{id}` - Stop job (already in job-manager)
- `GET /api/training/history` - Training history (could migrate)

**Keep**: Training start (subprocess)  
**Could migrate**: Validate, status, stop, history

### 2. LoRA Utilities (Keep as subprocess)
**File**: `api/routes/utilities.py` (LoRA section)

**Routes:**
- `POST /api/utilities/lora/resize` - Resize LoRA (uses safetensors + PyTorch)
- `POST /api/utilities/lora/merge` - Merge LoRAs (uses safetensors + PyTorch)
- `POST /api/utilities/checkpoint/merge` - Merge checkpoints (uses safetensors)

**Why keep Python**: 
- Requires `safetensors` library (no mature Node.js equivalent)
- Needs PyTorch for tensor manipulation
- Complex model surgery

**How**: Spawn Python script via job-manager (like training)

### 3. HuggingFace Upload (Keep as subprocess)
**File**: `api/routes/utilities.py` (HF section)

**Routes:**
- `POST /api/utilities/hf/upload` - Upload to HuggingFace Hub
- `POST /api/utilities/hf/validate-token` - Validate HF token

**Why keep Python**:
- `huggingface_hub` Python library is mature
- Node.js equivalent `@huggingface/hub` is less mature
- Already works well

**Could migrate**: Token validation (just HTTP call)  
**Keep subprocess**: Upload

### 4. Captioning (Already Python subprocess - working)
**File**: `api/routes/dataset.py` (caption section)

**Routes:**
- `POST /api/dataset/caption/blip` - BLIP captioning ✅
- `POST /api/dataset/caption/git` - GIT captioning ✅
- `GET /api/dataset/caption/status/{id}` - Status (already in job-manager)
- `POST /api/dataset/caption/stop/{id}` - Stop (already in job-manager)

**Status**: Already integrated with job-manager, working fine ✅

---

## 🎯 Recommended Migration Priority

### High Priority (Do Next)
1. **Config API routes** - Service exists, just need routes (2-3 hours)
   - Completes the config system
   - Critical for training workflow

### Medium Priority (Nice to Have)
2. **Dataset CRUD operations** - File uploads, ZIP extraction (4-6 hours)
   - Completes dataset management
   - User-facing features

3. **Model downloads** - Download orchestration (3-4 hours)
   - User-facing feature
   - Currently works but could be cleaner in Node.js

### Low Priority (Optional)
4. **Civitai integration** - Just HTTP calls (2-3 hours)
   - Works fine in Python currently
   - Low user impact

5. **Utilities (calculator, etc.)** - Simple logic (2-3 hours)
   - Works fine in Python
   - Low priority

### Don't Migrate (Keep Python)
- ❌ Training (Kohya SS) - Already subprocess ✅
- ❌ BLIP/GIT captioning - Already subprocess ✅
- ❌ LoRA resize/merge - Needs safetensors + PyTorch
- ❌ Checkpoint merge - Needs safetensors
- ❌ HuggingFace upload - Works well in Python

---

## 📋 Quick Win: Config API Routes

Since `config-service.ts` already exists, the quickest win is creating the API routes:

**Estimated time**: 2-3 hours  
**Benefit**: Completes config system, high user value

**Create:**
```
frontend/app/api/config/
├── templates/route.ts
├── load/route.ts
├── save/route.ts
├── save-training/route.ts
├── validate/route.ts
├── defaults/route.ts
└── presets/
    ├── route.ts (list, create)
    └── [id]/route.ts (get, delete)
```

---

## 📊 Migration Effort Estimate

| Phase | Routes | Effort | Priority | Status |
|-------|--------|--------|----------|--------|
| **Config API** | 10 | 2-3h | High | Ready (service exists) |
| **Dataset CRUD** | 8 | 4-6h | Medium | Moderate complexity |
| **Model Downloads** | 6 | 3-4h | Medium | Subprocess spawning |
| **Civitai** | 5 | 2-3h | Low | Simple HTTP |
| **Utilities** | 5 | 2-3h | Low | Simple logic |
| **TOTAL** | 34 routes | 13-19h | - | - |

---

## 🎮 What Can Stay in Python?

**Total Python ML subprocesses (keep as-is):**
- Training (Kohya SS)
- BLIP captioning
- GIT captioning
- JoyCaption (future)
- LoRA resize/merge
- Checkpoint merge
- HuggingFace upload

**These are FINE as Python subprocesses** because:
1. No asyncio issues (only WD14 tagging had problems)
2. Require ML libraries only in Python
3. Already integrated with job-manager.ts
4. Less frequently used than core features

---

## 🚀 Next Steps

### Immediate (This Week)
1. Test WD14 Node.js tagger on VastAI
2. Verify all Phase 1-5 migrations work

### Next Migration Session
1. **Create config API routes** (high priority, quick win)
2. **Test config system** end-to-end
3. **Decide on dataset CRUD** (medium priority, moderate effort)

### Future (Optional)
- Model downloads (if users request it)
- Civitai integration (if users request it)
- Utilities migration (low priority)

**Don't migrate**: Python ML subprocesses (working fine!)

---

**Want to tackle config API routes next?** It's the highest value, lowest effort task remaining! 🎯
