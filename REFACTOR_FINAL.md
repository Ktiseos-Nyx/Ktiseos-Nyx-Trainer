# 🚀 API-First Refactor - The Actual Plan

**Decision:** Scrap Jupyter managers, build clean API layer from scratch

**Why:** Managers were designed for notebooks, not web APIs. Fighting them is wasting time.

**Timeline:** 10-12 days to working alpha

---

## 🎯 Core Principles

### What We're Building:
- ✅ **Clean API layer** - Designed for FastAPI from day one
- ✅ **WebSocket log streaming** - Real-time training logs to frontend
- ✅ **Proper validation** - Pydantic models, path checking
- ✅ **Good error handling** - Frontend gets useful errors
- ✅ **Portable** - Works on VastAI, local, any deployment

### What We're NOT Building:
- ❌ Enterprise security theater
- ❌ Rate limiting (single-user instances)
- ❌ Redis/Celery (in-memory is fine)
- ❌ Auth system (not needed for VastAI)
- ❌ Multi-tenancy

### What We're Reusing:
- ✅ **TOML generation logic** - Extract from managers (it works!)
- ✅ **Model detection** - Copy the working patterns
- ✅ **Vendored code** - Kohya/Derrian untouched
- ❌ **Manager classes** - Leave them for notebooks, build new for API

---

## 📁 New Structure

```
services/
├── __init__.py
├── validation.py              # Path validation, file checking
├── exceptions.py              # Custom exception types
├── job_manager.py             # In-memory job tracking
├── models/                    # Pydantic models
│   ├── __init__.py
│   ├── common.py             # Shared models
│   ├── training.py           # Training config models
│   ├── dataset.py            # Dataset models
│   └── tagging.py            # Tagging models
├── dataset_service.py         # Dataset CRUD, upload, management
├── tagging_service.py         # WD14 tagging with job tracking
├── caption_service.py         # Caption editing operations
├── training_service.py        # LoRA training orchestration
└── model_service.py           # Model download/management

api/routes/
├── datasets.py               # Dataset endpoints
├── tagging.py                # Tagging endpoints
├── captions.py               # Caption endpoints
├── training.py               # Training endpoints
├── models.py                 # Model endpoints
└── websocket.py              # WebSocket log streaming
```

---

## 🔥 Phase 1: Foundation (Day 1-2)

### 1.1 Validation & Exceptions

```python
# services/validation.py
from pathlib import Path
from typing import Literal

# Base directories
DATASETS_DIR = Path("datasets").resolve()
MODELS_DIR = Path("pretrained_model").resolve()
OUTPUT_DIR = Path("output").resolve()

def validate_dataset_path(dataset_name: str) -> Path:
    """Validate dataset path is within datasets directory"""
    # Remove path traversal attempts
    clean_name = dataset_name.replace("..", "").strip("/")

    dataset_path = DATASETS_DIR / clean_name

    # Ensure it resolves within datasets directory
    if not str(dataset_path.resolve()).startswith(str(DATASETS_DIR)):
        raise ValueError(f"Invalid dataset path: {dataset_name}")

    return dataset_path

def validate_image_file(filename: str) -> str:
    """Validate image filename"""
    allowed_exts = {".jpg", ".jpeg", ".png", ".webp"}
    ext = Path(filename).suffix.lower()

    if ext not in allowed_exts:
        raise ValueError(f"Invalid file type: {ext}")

    # Return just the filename (no path components)
    return Path(filename).name

def validate_model_path(model_name: str) -> Path:
    """Validate model path is within models directory"""
    clean_name = model_name.replace("..", "").strip("/")
    model_path = MODELS_DIR / clean_name

    if not str(model_path.resolve()).startswith(str(MODELS_DIR)):
        raise ValueError(f"Invalid model path: {model_name}")

    return model_path
```

```python
# services/exceptions.py
class ServiceError(Exception):
    """Base exception for service layer"""
    pass

class ValidationError(ServiceError):
    """Invalid input data"""
    pass

class NotFoundError(ServiceError):
    """Resource not found"""
    pass

class ProcessError(ServiceError):
    """Subprocess/training execution failed"""
    pass

class ConfigError(ServiceError):
    """Invalid configuration"""
    pass
```

### 1.2 Job Manager with WebSocket Support

```python
# services/job_manager.py
from dataclasses import dataclass, field
from typing import Optional, AsyncIterator
from uuid import uuid4
import asyncio
from datetime import datetime

@dataclass
class Job:
    job_id: str
    type: str  # "tagging" | "training"
    status: str  # "running" | "completed" | "failed"
    created_at: datetime
    process: Optional[asyncio.subprocess.Process] = None
    error: Optional[str] = None

    # Progress tracking
    progress: int = 0  # 0-100
    current_step: Optional[str] = None

    # Log buffer for WebSocket streaming
    logs: list[str] = field(default_factory=list)
    max_logs: int = 1000  # Keep last 1000 lines

class JobManager:
    def __init__(self):
        self._jobs: dict[str, Job] = {}

    def create_job(self, job_type: str, process: asyncio.subprocess.Process) -> str:
        """Create new job and start monitoring"""
        job_id = str(uuid4())
        job = Job(
            job_id=job_id,
            type=job_type,
            status="running",
            created_at=datetime.now(),
            process=process
        )
        self._jobs[job_id] = job

        # Start log monitoring in background
        asyncio.create_task(self._monitor_logs(job_id))

        return job_id

    async def _monitor_logs(self, job_id: str):
        """Monitor process logs and update job"""
        job = self._jobs.get(job_id)
        if not job or not job.process:
            return

        try:
            # Read stdout line by line
            async for line in job.process.stdout:
                log_line = line.decode('utf-8').strip()

                # Add to log buffer (keep last max_logs)
                job.logs.append(log_line)
                if len(job.logs) > job.max_logs:
                    job.logs.pop(0)

                # Parse for progress info (Kohya outputs epoch info)
                self._parse_progress(job, log_line)

            # Process finished
            returncode = await job.process.wait()
            if returncode == 0:
                job.status = "completed"
                job.progress = 100
            else:
                job.status = "failed"
                job.error = f"Process exited with code {returncode}"

        except Exception as e:
            job.status = "failed"
            job.error = str(e)

    def _parse_progress(self, job: Job, log_line: str):
        """Parse Kohya logs for progress updates"""
        # Example: "epoch 3/10, step 150/500"
        if "epoch" in log_line.lower():
            # Extract epoch numbers
            import re
            match = re.search(r'epoch\s+(\d+)/(\d+)', log_line, re.I)
            if match:
                current, total = int(match.group(1)), int(match.group(2))
                job.progress = int((current / total) * 100)
                job.current_step = f"Epoch {current}/{total}"

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        return self._jobs.get(job_id)

    async def stream_logs(self, job_id: str, start_line: int = 0) -> AsyncIterator[str]:
        """Stream logs for WebSocket"""
        job = self._jobs.get(job_id)
        if not job:
            return

        # Send existing logs first
        for line in job.logs[start_line:]:
            yield line

        # Then stream new logs as they arrive
        last_line = len(job.logs)
        while job.status == "running":
            await asyncio.sleep(0.1)  # Check every 100ms

            # Yield new lines
            for line in job.logs[last_line:]:
                yield line
            last_line = len(job.logs)

# Global singleton
job_manager = JobManager()
```

### 1.3 Common Pydantic Models

```python
# services/models/common.py
from pydantic import BaseModel
from typing import Optional, Generic, TypeVar
from datetime import datetime

T = TypeVar('T')

class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response"""
    items: list[T]
    total: int
    page: int
    page_size: int

class JobStatus(BaseModel):
    """Job status response"""
    job_id: str
    status: str
    progress: int
    current_step: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: Optional[str] = None
```

**Deliverables:**
- [ ] `services/validation.py` (50 lines)
- [ ] `services/exceptions.py` (20 lines)
- [ ] `services/job_manager.py` (150 lines)
- [ ] `services/models/common.py` (30 lines)

---

## 📦 Phase 2: Dataset Service (Day 3-4)

```python
# services/models/dataset.py
from pydantic import BaseModel, Field
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
```

```python
# services/dataset_service.py
from pathlib import Path
from typing import List
from fastapi import UploadFile
import shutil

from .validation import validate_dataset_path, validate_image_file
from .exceptions import NotFoundError, ValidationError
from .models.dataset import Dataset, ImageInfo, UploadResponse

class DatasetService:
    """Dataset management operations"""

    async def list_datasets(self, page: int = 1, page_size: int = 20) -> dict:
        """List all datasets with pagination"""
        from .validation import DATASETS_DIR

        if not DATASETS_DIR.exists():
            return {"items": [], "total": 0, "page": page, "page_size": page_size}

        # Get all dataset directories
        all_datasets = [
            Dataset(
                name=d.name,
                path=str(d),
                image_count=len(list(d.glob("*.{jpg,jpeg,png,webp}"))),
                created_at=str(d.stat().st_ctime)
            )
            for d in DATASETS_DIR.iterdir()
            if d.is_dir()
        ]

        # Paginate
        start = (page - 1) * page_size
        end = start + page_size

        return {
            "items": all_datasets[start:end],
            "total": len(all_datasets),
            "page": page,
            "page_size": page_size
        }

    async def create_dataset(self, name: str) -> Dataset:
        """Create new dataset directory"""
        dataset_path = validate_dataset_path(name)

        if dataset_path.exists():
            raise ValidationError(f"Dataset already exists: {name}")

        dataset_path.mkdir(parents=True, exist_ok=True)

        return Dataset(
            name=name,
            path=str(dataset_path),
            image_count=0
        )

    async def upload_images(
        self,
        dataset_name: str,
        files: List[UploadFile]
    ) -> UploadResponse:
        """Upload images to dataset"""
        dataset_path = validate_dataset_path(dataset_name)

        if not dataset_path.exists():
            raise NotFoundError(f"Dataset not found: {dataset_name}")

        uploaded = 0
        failed = 0
        errors = []

        for file in files:
            try:
                # Validate filename
                safe_name = validate_image_file(file.filename)
                file_path = dataset_path / safe_name

                # Save file
                with file_path.open("wb") as f:
                    shutil.copyfileobj(file.file, f)

                uploaded += 1

            except Exception as e:
                failed += 1
                errors.append(f"{file.filename}: {str(e)}")

        return UploadResponse(
            uploaded=uploaded,
            failed=failed,
            errors=errors
        )

    async def get_dataset_images(
        self,
        dataset_name: str,
        page: int = 1,
        page_size: int = 50
    ) -> dict:
        """Get images in dataset with pagination"""
        dataset_path = validate_dataset_path(dataset_name)

        if not dataset_path.exists():
            raise NotFoundError(f"Dataset not found: {dataset_name}")

        # Get all images
        image_exts = {".jpg", ".jpeg", ".png", ".webp"}
        all_images = [
            ImageInfo(
                filename=img.name,
                path=str(img),
                caption_path=str(img.with_suffix(".txt")) if img.with_suffix(".txt").exists() else None,
                has_caption=img.with_suffix(".txt").exists()
            )
            for img in dataset_path.iterdir()
            if img.suffix.lower() in image_exts
        ]

        # Paginate
        start = (page - 1) * page_size
        end = start + page_size

        return {
            "items": all_images[start:end],
            "total": len(all_images),
            "page": page,
            "page_size": page_size
        }

    async def delete_dataset(self, dataset_name: str) -> bool:
        """Delete dataset directory"""
        dataset_path = validate_dataset_path(dataset_name)

        if not dataset_path.exists():
            raise NotFoundError(f"Dataset not found: {dataset_name}")

        shutil.rmtree(dataset_path)
        return True
```

**API Routes:**

```python
# api/routes/datasets.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List

from services.dataset_service import DatasetService
from services.exceptions import NotFoundError, ValidationError

router = APIRouter(prefix="/datasets", tags=["datasets"])
service = DatasetService()

@router.get("/")
async def list_datasets(page: int = 1, page_size: int = 20):
    """List all datasets"""
    return await service.list_datasets(page, page_size)

@router.post("/")
async def create_dataset(name: str):
    """Create new dataset"""
    try:
        return await service.create_dataset(name)
    except ValidationError as e:
        raise HTTPException(400, detail=str(e))

@router.post("/{name}/upload")
async def upload_images(name: str, files: List[UploadFile] = File(...)):
    """Upload images to dataset"""
    try:
        return await service.upload_images(name, files)
    except NotFoundError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@router.get("/{name}/images")
async def get_images(name: str, page: int = 1, page_size: int = 50):
    """Get images in dataset"""
    try:
        return await service.get_dataset_images(name, page, page_size)
    except NotFoundError as e:
        raise HTTPException(404, detail=str(e))

@router.delete("/{name}")
async def delete_dataset(name: str):
    """Delete dataset"""
    try:
        await service.delete_dataset(name)
        return {"message": "Dataset deleted"}
    except NotFoundError as e:
        raise HTTPException(404, detail=str(e))
```

**Deliverables:**
- [ ] `services/models/dataset.py`
- [ ] `services/dataset_service.py`
- [ ] `api/routes/datasets.py`
- [ ] Test dataset CRUD operations

---

## 🏷️ Phase 3: Tagging Service (Day 5-6)

**Extract WD14 tagging logic, add job tracking:**

```python
# services/models/tagging.py
from pydantic import BaseModel, Field
from typing import Optional

class TaggingRequest(BaseModel):
    dataset_name: str
    model: str = "wd14-vit-v2"
    general_threshold: float = Field(0.35, ge=0.0, le=1.0)
    character_threshold: float = Field(0.35, ge=0.0, le=1.0)
    blacklist_tags: Optional[str] = None
    remove_underscore: bool = True
    batch_size: int = Field(4, ge=1, le=32)
```

```python
# services/tagging_service.py
import sys
import asyncio
from pathlib import Path

from .validation import validate_dataset_path
from .job_manager import job_manager
from .models.tagging import TaggingRequest
from .exceptions import NotFoundError

class TaggingService:
    """WD14 tagging operations"""

    async def start_tagging(self, request: TaggingRequest) -> str:
        """Start tagging job"""
        # Validate dataset exists
        dataset_path = validate_dataset_path(request.dataset_name)
        if not dataset_path.exists():
            raise NotFoundError(f"Dataset not found: {request.dataset_name}")

        # Build command args (safe - no string concat!)
        script = Path("custom/tag_images_by_wd14_tagger.py")
        args = [
            sys.executable,
            str(script),
            str(dataset_path),
            "--model", request.model,
            "--general_threshold", str(request.general_threshold),
            "--character_threshold", str(request.character_threshold),
            "--batch_size", str(request.batch_size),
        ]

        if request.blacklist_tags:
            args.extend(["--blacklist_tags", request.blacklist_tags])

        if request.remove_underscore:
            args.append("--remove_underscore")

        # Start subprocess
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Create job
        job_id = job_manager.create_job("tagging", proc)

        return job_id
```

**WebSocket endpoint for logs:**

```python
# api/routes/websocket.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from services.job_manager import job_manager

router = APIRouter()

@router.websocket("/ws/logs/{job_id}")
async def stream_logs(websocket: WebSocket, job_id: str):
    """Stream job logs via WebSocket"""
    await websocket.accept()

    try:
        async for log_line in job_manager.stream_logs(job_id):
            await websocket.send_text(log_line)

        # Job finished
        await websocket.close()

    except WebSocketDisconnect:
        pass
```

**Deliverables:**
- [ ] `services/models/tagging.py`
- [ ] `services/tagging_service.py`
- [ ] `api/routes/tagging.py`
- [ ] `api/routes/websocket.py` (WebSocket log streaming)
- [ ] Test tagging with small dataset

---

## ✍️ Phase 4: Caption Service (Day 7)

**Simple file operations for caption editing:**

```python
# services/caption_service.py
from pathlib import Path
from typing import List

from .validation import validate_dataset_path
from .exceptions import NotFoundError

class CaptionService:
    """Caption file operations"""

    async def get_caption(self, dataset_name: str, image_name: str) -> str:
        """Get caption for image"""
        dataset_path = validate_dataset_path(dataset_name)
        image_path = dataset_path / image_name
        caption_path = image_path.with_suffix(".txt")

        if not caption_path.exists():
            return ""

        return caption_path.read_text()

    async def update_caption(self, dataset_name: str, image_name: str, caption: str):
        """Update caption file"""
        dataset_path = validate_dataset_path(dataset_name)
        image_path = dataset_path / image_name
        caption_path = image_path.with_suffix(".txt")

        caption_path.write_text(caption)
        return True

    async def inject_trigger_word(
        self,
        dataset_name: str,
        trigger_word: str,
        position: str = "start"
    ) -> int:
        """Inject trigger word into all captions"""
        dataset_path = validate_dataset_path(dataset_name)

        modified = 0
        for caption_file in dataset_path.glob("*.txt"):
            content = caption_file.read_text().strip()

            if position == "start":
                new_content = f"{trigger_word}, {content}"
            elif position == "end":
                new_content = f"{content}, {trigger_word}"
            else:  # both
                new_content = f"{trigger_word}, {content}, {trigger_word}"

            caption_file.write_text(new_content)
            modified += 1

        return modified
```

**Deliverables:**
- [ ] `services/caption_service.py`
- [ ] `api/routes/captions.py`

---

## 🎓 Phase 5: Training Service (Day 8-10)

**This is the big one - extract TOML generation from managers:**

```python
# services/training_service.py
import sys
import asyncio
import toml
from pathlib import Path

from .validation import validate_dataset_path, validate_model_path
from .job_manager import job_manager
from .exceptions import NotFoundError, ConfigError

class TrainingService:
    """LoRA training operations"""

    # Copy from KohyaTrainingManager
    SCRIPT_MAPPING = {
        'sd15': 'train_network.py',
        'sdxl': 'sdxl_train_network.py',
        'flux': 'flux_train_network.py',
    }

    async def start_training(self, config: dict) -> str:
        """Start training job"""
        # Validate paths
        dataset_path = validate_dataset_path(config['dataset_path'])
        if not dataset_path.exists():
            raise NotFoundError(f"Dataset not found: {config['dataset_path']}")

        # Generate TOML configs (COPY from KohyaTrainingManager)
        config_file = self._generate_config_toml(config)
        dataset_file = self._generate_dataset_toml(config)

        # Get training script
        script = self.SCRIPT_MAPPING.get(config['model_type'])
        if not script:
            raise ConfigError(f"Unknown model type: {config['model_type']}")

        # Build command
        script_path = Path(f"trainer/derrian_backend/sd_scripts/{script}")
        args = [
            sys.executable,
            str(script_path),
            "--config_file", str(config_file),
            "--dataset_config", str(dataset_file)
        ]

        # Start training
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Create job
        job_id = job_manager.create_job("training", proc)

        return job_id

    def _generate_config_toml(self, config: dict) -> Path:
        """
        Generate training config TOML

        TODO: COPY logic from KohyaTrainingManager.create_config_toml()
        This is already working - just extract it!
        """
        # Use your existing TOML generation logic
        pass

    def _generate_dataset_toml(self, config: dict) -> Path:
        """
        Generate dataset config TOML

        TODO: COPY logic from KohyaTrainingManager.create_dataset_toml()
        """
        pass
```

**Deliverables:**
- [ ] Extract TOML generation from `core/kohya_training_manager.py`
- [ ] `services/training_service.py`
- [ ] `api/routes/training.py`
- [ ] Test training with small dataset (1 epoch)

---

## 🎨 Phase 6: Frontend (Day 11-12)

**Next.js with shadcn/ui + WebSocket logs:**

```typescript
// frontend/hooks/useWebSocket.ts
import { useEffect, useState } from 'react';

export function useJobLogs(jobId: string | null) {
  const [logs, setLogs] = useState<string[]>([]);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    if (!jobId) return;

    const ws = new WebSocket(`ws://localhost:8000/ws/logs/${jobId}`);

    ws.onopen = () => setConnected(true);
    ws.onmessage = (event) => {
      setLogs(prev => [...prev, event.data]);
    };
    ws.onclose = () => setConnected(false);

    return () => ws.close();
  }, [jobId]);

  return { logs, connected };
}
```

```typescript
// frontend/app/training/page.tsx
'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useJobLogs } from '@/hooks/useWebSocket';

export default function TrainingPage() {
  const [jobId, setJobId] = useState<string | null>(null);
  const { logs, connected } = useJobLogs(jobId);

  async function startTraining() {
    const res = await fetch('/api/training/start', {
      method: 'POST',
      body: JSON.stringify({ /* config */ })
    });
    const { job_id } = await res.json();
    setJobId(job_id);
  }

  return (
    <div>
      <Button onClick={startTraining}>Start Training</Button>

      {jobId && (
        <div>
          <h2>Training Logs</h2>
          <ScrollArea className="h-96 border p-4">
            {logs.map((log, i) => (
              <div key={i} className="font-mono text-sm">{log}</div>
            ))}
          </ScrollArea>
        </div>
      )}
    </div>
  );
}
```

**Deliverables:**
- [ ] Dataset management UI
- [ ] Tagging interface
- [ ] Training config form
- [ ] Live log viewer with WebSocket

---

## ✅ Success Criteria

**Alpha is ready when:**
- ✅ Can create dataset via web UI
- ✅ Can upload images via web UI
- ✅ Can start tagging via web UI
- ✅ Can see tagging logs in real-time (WebSocket)
- ✅ Can configure training via web UI
- ✅ Can start training via web UI
- ✅ Can see training logs in real-time (WebSocket)
- ✅ Works on VastAI container
- ✅ Works on local machine
- ✅ No critical bugs

---

## 📊 What We're Building vs What We're Not

| Feature | Status | Why |
|---------|--------|-----|
| **Services layer** | ✅ YES | Clean API, testable |
| **Pydantic models** | ✅ YES | Validation needed |
| **WebSocket logs** | ✅ YES | Real-time feedback needed |
| **Path validation** | ✅ YES | Prevents crashes |
| **Error handling** | ✅ YES | Frontend needs it |
| **Job tracking** | ✅ YES (in-memory) | Progress needed |
| **Auth system** | ❌ NO | Single-user instances |
| **Rate limiting** | ❌ NO | Single-user instances |
| **Redis/Celery** | ❌ NO | In-memory is fine |
| **Database** | ❌ NO | Files + in-memory |
| **Security audit** | ❌ NO | Not public SaaS |

---

## 🎯 Timeline

- **Days 1-2:** Foundation (validation, exceptions, job manager)
- **Days 3-4:** Dataset service
- **Days 5-6:** Tagging service + WebSocket
- **Day 7:** Caption service
- **Days 8-10:** Training service (extract TOML logic)
- **Days 11-12:** Frontend integration
- **Total: 12 days to alpha**

---

## 🚀 Next Steps

1. **Start Phase 1** - Build foundation (validation, job manager)
2. **Test early** - Don't wait until end to test
3. **Extract, don't rewrite** - Copy working TOML logic from managers
4. **Ship alpha** - Get it working, refine later

**Ready to build?**
