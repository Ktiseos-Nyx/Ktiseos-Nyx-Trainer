# 🏗️ V1 Architecture - Next.js LoRA Trainer

**Goal:** Next.js alternative to bmaltais/kohya_ss Gradio UI

**Philosophy:** Clean architecture, not enterprise. Built for single-user VastAI/local use, designed to extend later.

---

## 🎯 V1 Scope (Ship This First)

### **Core Features**
1. ✅ **Dataset Management**
   - Direct image upload (drag & drop, multi-file)
   - URL/ZIP download & extract
   - Image gallery browser

2. ✅ **Auto-Tagging**
   - WD14 tagger (multiple models, thresholds, blacklist)
   - WebSocket live progress
   - Job tracking

3. ✅ **Caption Editing**
   - Trigger word injection (start/end/both)
   - Search & replace tags
   - Remove tags
   - Individual caption editing

4. ✅ **Training**
   - Full Kohya config (all parameters)
   - Real-time step calculator (live in form)
   - Training execution
   - WebSocket live logs
   - Progress tracking (parse Kohya logs)
   - Stop/cancel training

5. ✅ **LoRA Utilities**
   - Upload to HuggingFace Hub
   - Resize LoRA (change dimension)
   - Browse trained LoRAs

6. ✅ **Model Management** (Basic UI)
   - List downloaded models
   - Download from Civitai (existing backend)
   - Download from HuggingFace (existing backend)

7. ✅ **Utilities**
   - Step calculator (standalone page)

### **Architecture (Built-in from Start)**
- ✅ **Pydantic models** - Validation for all configs
- ✅ **Service layer** - Clean separation
- ✅ **BaseTrainer pattern** - Extensible for future trainers
- ✅ **WebSocket streaming** - Real-time logs
- ✅ **Job tracking** - In-memory (simple dict)

---

## 🔮 Future Roadmap (v2+)

### **Performance & Hardware**
- ⏳ **Multi-GPU support** - Train across multiple GPUs
- ⏳ **ROCm/AMD support** - Not just NVIDIA
- ⏳ **Intel Arc support** - OneAPI

### **Training Features**
- ⏳ **Checkpoint training** - Resume from checkpoint, continue training existing LoRA
- ⏳ **Other model types** - Dreambooth, Textual Inversion, Hypernetworks, full fine-tuning
- ⏳ **Other trainer backends** - Not just Kohya (vendor more trainers)

### **Quality of Life**
- ⏳ **Job history** - SQLite storage, view past runs
- ⏳ **Config presets** - Save favorite settings
- ⏳ **WebSocket reconnection** - Auto-reconnect on connection drop
- ⏳ **Better error messages** - Contextual help

### **Advanced Features**
- ⏳ **Gallery-DL scraper** - Download datasets from sites
- ⏳ **Inference testing** - Test LoRAs after training
- ⏳ **Model comparison** - Compare training runs

---

## 📁 Project Structure

```
Ktiseos-Nyx-Trainer/
├── api/                          # FastAPI backend
│   ├── main.py                   # FastAPI app
│   └── routes/
│       ├── datasets.py           # Dataset CRUD, upload
│       ├── tagging.py            # WD14 tagging
│       ├── captions.py           # Caption editing
│       ├── training.py           # Training execution
│       ├── models.py             # Model downloads
│       ├── lora_utils.py         # LoRA utilities
│       └── websocket.py          # WebSocket log streaming
│
├── services/                     # Business logic layer
│   ├── __init__.py
│   ├── validation.py             # Path/file validation
│   ├── exceptions.py             # Custom exceptions
│   ├── job_manager.py            # Job tracking & WebSocket
│   │
│   ├── models/                   # Pydantic models
│   │   ├── __init__.py
│   │   ├── common.py             # Shared models
│   │   ├── dataset.py            # Dataset models
│   │   ├── tagging.py            # Tagging models
│   │   ├── training.py           # Training config models
│   │   └── lora.py               # LoRA utility models
│   │
│   ├── trainers/                 # Trainer implementations
│   │   ├── __init__.py
│   │   ├── base.py               # BaseTrainer (abstract)
│   │   └── kohya_trainer.py      # Kohya implementation
│   │
│   ├── dataset_service.py        # Dataset operations
│   ├── tagging_service.py        # WD14 tagging
│   ├── caption_service.py        # Caption editing
│   ├── training_service.py       # Training orchestration
│   ├── model_service.py          # Model downloads
│   └── lora_service.py           # LoRA utilities
│
├── frontend/                     # Next.js app
│   ├── app/
│   │   ├── page.tsx              # Home/dashboard
│   │   ├── datasets/
│   │   │   ├── page.tsx          # Dataset list
│   │   │   └── [name]/
│   │   │       ├── page.tsx      # Image gallery
│   │   │       └── tag/page.tsx  # Tagging interface
│   │   ├── training/
│   │   │   ├── page.tsx          # Training config
│   │   │   └── [jobId]/page.tsx  # Training monitor
│   │   ├── models/page.tsx       # Model browser
│   │   ├── lora-utils/page.tsx   # LoRA utilities
│   │   └── calculator/page.tsx   # Step calculator
│   │
│   ├── components/
│   │   ├── ui/                   # shadcn/ui components
│   │   ├── dataset/              # Dataset-specific
│   │   ├── training/             # Training-specific
│   │   └── shared/               # Shared components
│   │
│   ├── hooks/
│   │   ├── useWebSocket.ts       # WebSocket hook
│   │   ├── useJobStatus.ts       # Job polling
│   │   └── useStepCalc.ts        # Step calculator
│   │
│   └── lib/
│       ├── api.ts                # API client
│       └── utils.ts              # Helpers
│
├── trainer/                      # Vendored backends
│   └── derrian_backend/          # Kohya + Derrian (don't touch)
│       ├── sd_scripts/           # Kohya scripts
│       └── lycoris/              # LyCORIS
│
├── custom/                       # Custom scripts
│   └── tag_images_by_wd14_tagger.py
│
├── datasets/                     # User datasets
├── pretrained_model/             # Downloaded models
├── output/                       # Trained LoRAs
└── training_configs/             # Generated TOML files
```

---

## 🔧 Service Layer Architecture

### **Base Trainer Pattern** (Extensibility)

```python
# services/trainers/base.py
from abc import ABC, abstractmethod
from pydantic import BaseModel

class BaseTrainer(ABC):
    """
    Base class for all trainer implementations.

    Future trainers inherit from this:
    - KohyaTrainer (v1)
    - DreamboothTrainer (future)
    - OtherBackendTrainer (future)
    """

    @abstractmethod
    async def start_training(
        self,
        config: "TrainingConfig",
        resume_from: Optional[str] = None  # For checkpoint training (future)
    ) -> str:
        """
        Start training job, return job_id

        Args:
            config: Training configuration
            resume_from: Optional checkpoint path to resume from

        Returns:
            job_id: Unique job identifier
        """
        pass

    @abstractmethod
    def validate_config(self, config: "TrainingConfig") -> None:
        """
        Validate config for this trainer
        Raises ValidationError if invalid
        """
        pass

    @abstractmethod
    def generate_config_files(self, config: "TrainingConfig") -> List[Path]:
        """
        Generate trainer-specific config files (TOML, JSON, etc.)
        Returns list of generated file paths
        """
        pass
```

### **Kohya Trainer Implementation** (v1)

```python
# services/trainers/kohya_trainer.py
from .base import BaseTrainer
import asyncio
import toml
from pathlib import Path

class KohyaTrainer(BaseTrainer):
    """
    Kohya-ss trainer implementation
    Uses vendored trainer/derrian_backend/sd_scripts
    """

    SCRIPT_MAPPING = {
        'sd15': 'train_network.py',
        'sd20': 'train_network.py',
        'sdxl': 'sdxl_train_network.py',
        'flux': 'flux_train_network.py',
        'sd3': 'sd3_train_network.py',
    }

    async def start_training(
        self,
        config: TrainingConfig,
        resume_from: Optional[str] = None
    ) -> str:
        # Validate config
        self.validate_config(config)

        # Generate TOML files
        config_files = self.generate_config_files(config)

        # Get appropriate Kohya script
        script = self._get_script(config.model_type)

        # Build args
        args = [
            sys.executable,
            f"trainer/derrian_backend/sd_scripts/{script}",
            "--config_file", str(config_files[0]),
            "--dataset_config", str(config_files[1]),
        ]

        # Add resume flag if provided
        if resume_from:
            args.extend(["--resume", resume_from])

        # Start subprocess
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Create job with job manager
        from services.job_manager import job_manager
        job_id = job_manager.create_job("training", proc)

        return job_id

    def validate_config(self, config: TrainingConfig) -> None:
        """Validate Kohya-specific requirements"""
        from services.validation import validate_dataset_path, validate_model_path

        # Validate paths exist
        dataset_path = validate_dataset_path(config.dataset_name)
        if not dataset_path.exists():
            raise NotFoundError(f"Dataset not found: {config.dataset_name}")

        model_path = validate_model_path(config.model_name)
        if not model_path.exists():
            raise NotFoundError(f"Model not found: {config.model_name}")

        # Validate parameter ranges
        if not 0 < config.learning_rate < 1:
            raise ValidationError("Learning rate must be between 0 and 1")

        # ... more validations

    def generate_config_files(self, config: TrainingConfig) -> List[Path]:
        """
        Generate Kohya TOML files
        Returns [config.toml, dataset.toml]
        """
        config_toml = self._generate_config_toml(config)
        dataset_toml = self._generate_dataset_toml(config)

        return [config_toml, dataset_toml]

    def _generate_config_toml(self, config: TrainingConfig) -> Path:
        """Generate training config TOML"""
        toml_config = {
            "Basics": {
                "pretrained_model_name_or_path": config.model_path,
                "resolution": config.resolution,
                "seed": config.seed,
                "max_train_epochs": config.epochs,
                "clip_skip": config.clip_skip,
            },
            "Save": {
                "output_dir": "output",
                "output_name": config.output_name,
                "save_precision": config.save_precision,
                "save_model_as": "safetensors",
                "save_every_n_epochs": config.save_every_n_epochs,
            },
            "Network_setup": {
                "network_dim": config.network_dim,
                "network_alpha": config.network_alpha,
            },
            "Optimizer": {
                "train_batch_size": config.batch_size,
                "optimizer_type": config.optimizer,
                "unet_lr": config.learning_rate,
                "text_encoder_lr": config.text_encoder_lr,
            },
            # ... all other sections
        }

        output_path = Path(f"training_configs/{config.output_name}.toml")
        with open(output_path, 'w') as f:
            toml.dump(toml_config, f)

        return output_path

    def _generate_dataset_toml(self, config: TrainingConfig) -> Path:
        """Generate dataset config TOML"""
        # ... dataset TOML generation
        pass

    def _get_script(self, model_type: str) -> str:
        """Get Kohya script for model type"""
        return self.SCRIPT_MAPPING.get(model_type, 'train_network.py')
```

### **Training Service** (Orchestration)

```python
# services/training_service.py
from .trainers.kohya_trainer import KohyaTrainer
from .trainers.base import BaseTrainer

class TrainingService:
    """
    High-level training orchestration
    Delegates to specific trainer implementations
    """

    def __init__(self):
        # v1: Only Kohya
        self.trainers = {
            "kohya": KohyaTrainer()
        }

        # Future: Add more trainers
        # self.trainers["dreambooth"] = DreamboothTrainer()
        # self.trainers["other"] = OtherTrainer()

    async def start_training(
        self,
        config: TrainingConfig,
        trainer_type: str = "kohya"
    ) -> str:
        """
        Start training with specified trainer

        Args:
            config: Training configuration
            trainer_type: Which trainer to use (default: kohya)

        Returns:
            job_id: Job identifier for tracking
        """
        trainer = self.trainers.get(trainer_type)
        if not trainer:
            raise ValueError(f"Unknown trainer: {trainer_type}")

        return await trainer.start_training(config)

    async def get_training_status(self, job_id: str) -> dict:
        """Get training job status"""
        from services.job_manager import job_manager
        return await job_manager.get_job_status(job_id)

    async def stop_training(self, job_id: str) -> bool:
        """Stop running training job"""
        from services.job_manager import job_manager
        return await job_manager.stop_job(job_id)
```

---

## 📊 Pydantic Models

### **Training Configuration** (Main Model)

```python
# services/models/training.py
from pydantic import BaseModel, Field
from typing import Optional, Literal

class TrainingConfig(BaseModel):
    """
    Complete training configuration
    Validated before training starts
    """

    # Project settings
    output_name: str = Field(..., min_length=1, description="LoRA output name")
    dataset_name: str = Field(..., description="Dataset directory name")
    model_name: str = Field(..., description="Base model filename")

    # Basic training
    resolution: int = Field(1024, ge=256, le=2048, description="Training resolution")
    epochs: int = Field(10, ge=1, le=1000, description="Number of epochs")
    batch_size: int = Field(1, ge=1, le=32, description="Batch size")

    # Learning rates
    learning_rate: float = Field(1e-4, gt=0, lt=1, description="UNet learning rate")
    text_encoder_lr: Optional[float] = Field(None, gt=0, lt=1, description="Text encoder LR")

    # LoRA structure
    network_dim: int = Field(32, ge=4, le=256, description="LoRA rank")
    network_alpha: int = Field(16, ge=1, le=256, description="LoRA alpha")

    # Optimizer
    optimizer: str = Field("AdamW8bit", description="Optimizer type")
    lr_scheduler: str = Field("constant", description="LR scheduler")

    # Advanced
    clip_skip: int = Field(2, ge=1, le=12)
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    gradient_checkpointing: bool = False

    # Saving
    save_every_n_epochs: int = Field(1, ge=1)
    save_precision: Literal["float", "fp16", "bf16"] = "fp16"

    # Model type (auto-detected)
    model_type: Literal["sd15", "sd20", "sdxl", "flux", "sd3"] = "sdxl"

    # Future: GPU configuration
    # gpu_ids: List[int] = Field(default=[0], description="GPUs to use")
    # use_multi_gpu: bool = False

    # Future: Trainer selection
    # trainer_backend: str = "kohya"

    class Config:
        json_schema_extra = {
            "example": {
                "output_name": "my_character_lora",
                "dataset_name": "my_character_dataset",
                "model_name": "animagine-xl-3.1.safetensors",
                "resolution": 1024,
                "epochs": 10,
                "batch_size": 2,
                "learning_rate": 1e-4,
                "network_dim": 32,
                "network_alpha": 16,
            }
        }
```

### **Tagging Configuration**

```python
# services/models/tagging.py
from pydantic import BaseModel, Field
from typing import Optional

class TaggingRequest(BaseModel):
    """WD14 tagging configuration"""

    dataset_name: str = Field(..., description="Dataset to tag")
    model: str = Field("wd14-vit-v2", description="Tagger model")
    general_threshold: float = Field(0.35, ge=0.0, le=1.0)
    character_threshold: float = Field(0.35, ge=0.0, le=1.0)
    blacklist_tags: Optional[str] = Field(None, description="Comma-separated tags to exclude")
    remove_underscore: bool = True
    batch_size: int = Field(4, ge=1, le=32)
    caption_extension: str = Field(".txt", pattern=r"^\.(txt|caption)$")

    class Config:
        json_schema_extra = {
            "example": {
                "dataset_name": "my_character_dataset",
                "model": "wd14-swinv2-v3",
                "general_threshold": 0.35,
                "character_threshold": 0.85,
                "blacklist_tags": "simple_background,white_background",
            }
        }
```

### **Dataset Models**

```python
# services/models/dataset.py
from pydantic import BaseModel
from typing import Optional

class Dataset(BaseModel):
    name: str
    path: str
    image_count: int
    created_at: Optional[str] = None

class ImageInfo(BaseModel):
    filename: str
    path: str
    caption_path: Optional[str] = None
    has_caption: bool

class UploadResponse(BaseModel):
    uploaded: int
    failed: int
    errors: list[str] = []

class DatasetFromURLRequest(BaseModel):
    project_name: str = Field(..., min_length=1, pattern=r"^[a-zA-Z0-9_-]+$")
    url: str = Field(..., description="URL or file path to ZIP")
```

### **Common Models**

```python
# services/models/common.py
from pydantic import BaseModel
from typing import Generic, TypeVar
from datetime import datetime

T = TypeVar('T')

class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    page_size: int

class JobStatus(BaseModel):
    job_id: str
    status: str  # "running" | "completed" | "failed"
    progress: int  # 0-100
    current_step: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
```

---

## 🔌 API Routes

### **Training Routes**

```python
# api/routes/training.py
from fastapi import APIRouter, HTTPException
from services.training_service import TrainingService
from services.models.training import TrainingConfig
from services.exceptions import NotFoundError, ValidationError

router = APIRouter(prefix="/training", tags=["training"])
service = TrainingService()

@router.post("/start")
async def start_training(config: TrainingConfig):
    """
    Start training job

    Returns job_id for monitoring progress
    """
    try:
        job_id = await service.start_training(config)
        return {"job_id": job_id, "status": "started"}
    except ValidationError as e:
        raise HTTPException(400, detail=str(e))
    except NotFoundError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Training failed: {e}")

@router.get("/status/{job_id}")
async def get_training_status(job_id: str):
    """Get training job status and progress"""
    status = await service.get_training_status(job_id)
    if not status:
        raise HTTPException(404, "Job not found")
    return status

@router.post("/stop/{job_id}")
async def stop_training(job_id: str):
    """Stop running training job"""
    success = await service.stop_training(job_id)
    if not success:
        raise HTTPException(404, "Job not found")
    return {"message": "Training stopped"}
```

### **WebSocket Route**

```python
# api/routes/websocket.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from services.job_manager import job_manager

router = APIRouter()

@router.websocket("/ws/logs/{job_id}")
async def stream_logs(websocket: WebSocket, job_id: str):
    """
    Stream job logs in real-time

    Usage:
        const ws = new WebSocket('ws://localhost:8000/ws/logs/job-123')
        ws.onmessage = (event) => console.log(event.data)
    """
    await websocket.accept()

    try:
        async for log_line in job_manager.stream_logs(job_id):
            await websocket.send_text(log_line)

        # Job finished, close connection
        await websocket.close()

    except WebSocketDisconnect:
        pass
```

---

## 🎨 Frontend Architecture

### **Key Pages**

**1. Dashboard** (`/`)
- Quick stats
- Recent jobs
- Quick actions

**2. Dataset Management** (`/datasets`)
- List all datasets
- Upload new dataset
- Download from URL

**3. Dataset Detail** (`/datasets/[name]`)
- Image gallery
- Caption viewer
- Quick caption edit
- Link to tagging

**4. Tagging Interface** (`/datasets/[name]/tag`)
- Tagging configuration form
- Start tagging button
- Live log viewer (WebSocket)
- Progress indicator

**5. Training Config** (`/training`)
- Multi-step form or tabs
- Dataset selection
- Model selection
- All Kohya parameters
- **Real-time step calculator** (updates as you type)
- Start training button

**6. Training Monitor** (`/training/[jobId]`)
- Live log viewer (WebSocket)
- Progress bar (parsed from logs)
- Current epoch/step
- Stop button
- Training stats

**7. LoRA Utilities** (`/lora-utils`)
- Browse trained LoRAs
- Upload to HuggingFace
- Resize LoRA

**8. Step Calculator** (`/calculator`)
- Standalone calculator
- Parameter inputs
- Real-time calculations

### **Key Hooks**

```typescript
// frontend/hooks/useWebSocket.ts
export function useJobLogs(jobId: string | null) {
  const [logs, setLogs] = useState<string[]>([])
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    if (!jobId) return

    const ws = new WebSocket(`ws://localhost:8000/ws/logs/${jobId}`)

    ws.onopen = () => setConnected(true)
    ws.onmessage = (event) => {
      setLogs(prev => [...prev, event.data])
    }
    ws.onclose = () => setConnected(false)

    return () => ws.close()
  }, [jobId])

  return { logs, connected }
}
```

```typescript
// frontend/hooks/useStepCalc.ts
export function useStepCalculator(
  images: number,
  epochs: number,
  batchSize: number,
  repeats: number = 1
) {
  return useMemo(() => {
    const steps = Math.floor((images * repeats * epochs) / batchSize)
    const estimatedMinutes = steps * 0.12 // ~7s per step on RTX 3090

    return {
      totalSteps: steps,
      estimatedTime: formatTime(estimatedMinutes),
      stepsPerEpoch: Math.floor((images * repeats) / batchSize)
    }
  }, [images, epochs, batchSize, repeats])
}
```

---

## 🚀 Deployment

### **VastAI Container**
```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3.10 python3-pip

# Copy project
COPY . /app
WORKDIR /app

# Install dependencies
RUN pip install -r requirements-backend.txt

# Expose ports
EXPOSE 8000 3000

# Start services
CMD ["./start.sh"]
```

### **Local Development**
```bash
# Backend
cd api
python -m uvicorn main:app --reload --port 8000

# Frontend
cd frontend
npm run dev
```

---

## ✅ Success Criteria

**v1 is ready when:**
- ✅ Can upload dataset via drag & drop
- ✅ Can download dataset from URL/ZIP
- ✅ Can browse dataset images
- ✅ Can auto-tag with WD14, see live logs
- ✅ Can edit captions (trigger words, search/replace)
- ✅ Can configure training with all Kohya params
- ✅ Can see real-time step calculation in form
- ✅ Can start training, see live logs via WebSocket
- ✅ Can track training progress (epoch/step parsed from logs)
- ✅ Can stop/cancel training
- ✅ Can upload trained LoRA to HuggingFace
- ✅ Can resize LoRA
- ✅ Works on VastAI container
- ✅ Works on local machine

---

**Ready to build! 🚀**
