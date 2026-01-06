# Services Layer - Refactored Backend

Clean, modular service layer to replace old Jupyter notebook managers.

## 📁 Structure

```
services/
├── core/                          # Core utilities
│   ├── exceptions.py              # Exception hierarchy (30 lines)
│   ├── validation.py              # Path validation (150 lines)
│   └── log_parser.py              # Kohya/WD14 log parsing (196 lines)
├── models/                        # Pydantic models
│   ├── common.py                  # Shared models (40 lines)
│   ├── job.py                     # Job tracking (80 lines)
│   ├── training.py                # Training config (290 lines)
│   ├── tagging.py                 # Tagging config (75 lines)
│   ├── dataset.py                 # Dataset models (60 lines)
│   ├── caption.py                 # Caption models (70 lines)
│   └── lora.py                    # LoRA utility models (60 lines)
├── jobs/                          # Job management
│   ├── job.py                     # Job dataclass (87 lines)
│   ├── job_store.py               # In-memory storage (65 lines)
│   └── job_manager.py             # Job orchestration (259 lines)
├── trainers/                      # Training backends
│   ├── base.py                    # BaseTrainer ABC (166 lines)
│   ├── kohya.py                   # Kohya implementation (163 lines)
│   └── kohya_toml.py              # TOML generation (209 lines)
├── websocket.py                   # WebSocket routes (134 lines)
├── training_service.py            # Training orchestration (156 lines)
├── tagging_service.py             # WD14 tagging (195 lines)
├── dataset_service.py             # Dataset management (220 lines)
├── caption_service.py             # Caption editing (250 lines)
├── lora_service.py                # LoRA utilities (240 lines)
├── __init__.py                    # Service exports (50 lines)
└── EXAMPLE_API_ROUTE.py           # API integration example (111 lines)
```

**Total: ~3200 lines across 24 focused files**
Compare to: Old managers were 1660-2348 lines EACH in monolithic files.

## ✅ What's Complete (ALL SERVICES BUILT!)

### Foundation ✅
- [x] Exception hierarchy for meaningful errors
- [x] Path validation (prevent traversal, basic security)
- [x] Pydantic models (common, job, training, tagging, dataset, caption, lora)
- [x] Log parser for Kohya/WD14 progress extraction

### Job System ✅
- [x] Job dataclass with log buffering (last 1000 lines)
- [x] In-memory job store (dict-based, no database)
- [x] Job manager with subprocess monitoring
- [x] WebSocket log streaming (real-time)
- [x] WebSocket status updates (500ms intervals)

### Training System ✅
- [x] TrainingConfig Pydantic model (~140 fields)
- [x] BaseTrainer abstract pattern for extensibility
- [x] KohyaTrainer implementation
- [x] TOML generation (dataset.toml, config.toml)
- [x] TrainingService orchestration

### Tagging System ✅
- [x] TaggingConfig Pydantic model
- [x] WD14 tagger integration (custom + vendored)
- [x] ONNX runtime auto-detection
- [x] TaggingService with job tracking
- [x] Support for v3 models

### Dataset System ✅
- [x] DatasetService for file operations
- [x] Create/list/browse/delete datasets
- [x] File metadata (images, captions, sizes)
- [x] Path validation and safety checks

### Caption System ✅
- [x] CaptionService for editing operations
- [x] Add trigger words (start/end position)
- [x] Remove specific tags (bulk)
- [x] Replace text (with regex support)
- [x] Read/write individual captions

### LoRA Utilities ✅
- [x] LoRAService for utilities
- [x] LoRA resizing (change rank/alpha)
- [x] HuggingFace Hub upload
- [x] Automatic README generation
- [x] Metadata support

## 🎯 Key Features

### 1. Job Tracking
```python
from services.training_service import training_service

# Start training
response = await training_service.start_training(config)
job_id = response.job_id  # "job-abc12345"

# Get status
status = await training_service.get_status(job_id)
print(f"Progress: {status.progress}%")  # Real-time progress

# Stream logs via WebSocket
# ws://localhost:8000/ws/jobs/{job_id}/logs
```

### 2. Extensible Trainer Pattern
```python
# Current: Kohya trainer
from services.trainers import KohyaTrainer
trainer = KohyaTrainer(config)

# Future: Other trainers
from services.trainers import MultiGPUKohyaTrainer, ROCmTrainer
trainer = MultiGPUKohyaTrainer(config)  # Same interface!
```

### 3. Clean API Integration
```python
# OLD: api/routes/training.py
from shared_managers import get_training_manager
manager = get_training_manager()  # 1660-line monolith!
result = manager.start_training(config_dict)

# NEW: api/routes/training.py
from services.training_service import training_service
response = await training_service.start_training(config)
```

## 📋 Next Steps (API Integration)

### Phase 2: Update API Routes ⏳
1. **Update `api/routes/training.py`** - Use TrainingService
2. **Update `api/routes/dataset.py`** - Use DatasetService + TaggingService + CaptionService
3. **Update `api/routes/utilities.py`** - Use LoRAService
4. **Add WebSocket routes** - Include `services/websocket.py` in main app

### Phase 3: Frontend Integration
- Frontend already expects `job_id` in responses (check `api.ts`)
- Test WebSocket connections work
- Verify all existing API contracts still match

### Phase 4: Testing
- **Local Testing** (Mac - what you CAN test):
  - Dataset creation/browsing ✓
  - Caption editing ✓
  - LoRA resizing/upload ✓
  - WD14 tagging ✓ (if ONNX works)
  - Training config validation ✓
  - Job tracking system ✓

- **VastAI Testing** (GPU - actual training):
  - Full training workflow
  - Real-time log streaming
  - Progress monitoring
  - Multi-epoch training

### Phase 5: Cleanup
- Delete old managers:
  - `core/dataset_manager.py` (2000+ lines)
  - `core/kohya_training_manager.py` (2348 lines)
  - `shared_managers.py` (lazy loader)
- Remove unused imports
- Test on VastAI container

## 🔄 Migration Strategy

### For Each Service
1. Read old manager to understand WORKING logic
2. Extract algorithms (TOML gen, validation, etc.)
3. Build new service with clean interface
4. Create Pydantic models for API
5. Update API routes incrementally
6. Test against frontend expectations

### Parallel Work (Optional)
- Can build multiple services in parallel
- Each service is independent (TaggingService doesn't need TrainingService)
- Just share common utilities (jobs, core, models)

## 💡 Design Principles

1. **Small Files** - Each file <200 lines, easy to understand
2. **Pydantic First** - Data validation from the start
3. **Copy Ideas, Not Code** - Extract working logic, not old structure
4. **Async By Default** - Non-blocking subprocess execution
5. **Job Tracking Built-In** - All long operations return job_id
6. **Future-Proof** - BaseTrainer pattern for extensibility

## 🚀 Usage Examples

### Start Training
```python
from services.models.training import TrainingConfig
from services.training_service import training_service

config = TrainingConfig(
    project_name="my_lora",
    model_type="SDXL",
    pretrained_model_name_or_path="/path/to/model.safetensors",
    train_data_dir="/path/to/dataset",
    output_dir="/path/to/output",
    resolution=1024,
    num_repeats=10,
    max_train_epochs=10,
    # ... 130+ more fields
)

response = await training_service.start_training(config)
if response.success:
    print(f"Training started: {response.job_id}")
else:
    print(f"Errors: {response.validation_errors}")
```

### Monitor Progress (WebSocket)
```javascript
// Frontend: Connect to WebSocket
const ws = new WebSocket(`ws://localhost:8000/ws/jobs/${jobId}/logs`);

ws.onmessage = (event) => {
    console.log('Log:', event.data);
    // Display in UI
};
```

### Get Status (REST)
```python
status = await training_service.get_status("job-abc12345")

print(f"Status: {status.status}")           # running/completed/failed
print(f"Progress: {status.progress}%")       # 0-100
print(f"Epoch: {status.current_epoch}/{status.total_epochs}")
```

## 🎨 Architecture Benefits

**Before (Jupyter Managers):**
- Monolithic files (1600-2300 lines each)
- No job tracking (frontend polls?)
- Synchronous blocking code
- Tight coupling between UI and logic
- Hard to test, hard to maintain

**After (Service Layer):**
- Modular files (<200 lines each)
- Built-in job tracking with WebSocket
- Async subprocess execution
- Clean API boundary (FastAPI ↔ Service ↔ Trainer)
- Easy to test, easy to extend

## 📖 See Also
- `EXAMPLE_API_ROUTE.py` - How to integrate with FastAPI
- `frontend/lib/api.ts` - Frontend API client expectations
- `V1_ARCHITECTURE.md` - Overall architecture spec
- `CLAUDE_DEVELOPMENT_RULES.md` - Development principles

---

**Status**: Phase 1 Complete ✅
**Next**: Build remaining services (Tagging, Dataset, Caption, LoRA)
