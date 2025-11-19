# Next.js Architecture for Ktiseos-Nyx-Trainer

## Overview

This document outlines the architecture for migrating from Jupyter notebooks to a Next.js web application with FastAPI backend.

## Why Next.js?

- ✅ **Professional UI** with full design control
- ✅ **Works on remote GPU servers** (VastAI, RunPod) and local machines
- ✅ **Low memory overhead** (~100MB vs Gradio's ~500MB)
- ✅ **No client installation** - just open browser
- ✅ **TypeScript** for better code quality
- ✅ **Built-in file manager** - no need for Jupyter Lab

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              Next.js Frontend (Port 3000)            │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │   Training   │  │   Dataset    │  │   File    │ │
│  │   Manager    │  │   Manager    │  │  Manager  │ │
│  │              │  │              │  │           │ │
│  │ - Config UI  │  │ - Upload     │  │ - Browse  │ │
│  │ - Start/Stop │  │ - Tag editor │  │ - Upload  │ │
│  │ - Progress   │  │ - Preview    │  │ - Delete  │ │
│  │ - Logs       │  │ - Organize   │  │ - Edit    │ │
│  └──────────────┘  └──────────────┘  └───────────┘ │
│                                                      │
│  ┌──────────────────────────────────────────────┐  │
│  │         Utilities                             │  │
│  │  - Parameter Calculator                       │  │
│  │  - Model Resizer                              │  │
│  │  - HuggingFace Uploader                       │  │
│  └──────────────────────────────────────────────┘  │
│                                                      │
└───────────────────────────┬──────────────────────────┘
                            │ HTTP/WebSocket
                            ↓
┌─────────────────────────────────────────────────────┐
│           FastAPI Backend (Port 8000)                │
│                                                      │
│  Endpoints:                                          │
│  - POST /api/training/start                         │
│  - GET  /api/training/status                        │
│  - WS   /api/training/logs                          │
│  - POST /api/dataset/upload                         │
│  - POST /api/dataset/tag                            │
│  - GET  /api/files/list                             │
│  - POST /api/files/upload                           │
│  - DELETE /api/files/{path}                         │
│                                                      │
│  Managers (existing Python code):                   │
│  - KohyaTrainingManager                             │
│  - DatasetManager                                   │
│  - ConfigManager                                    │
│  - FileManager                                      │
│                                                      │
└───────────────────────────┬─────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────┐
│         Kohya SD-Scripts (Submodule)                 │
│         /workspace/sd-scripts/                       │
└─────────────────────────────────────────────────────┘
```

## File Manager Component

### Features

1. **File Browser**
   - Tree view of directories
   - File/folder icons
   - Search functionality
   - Breadcrumb navigation

2. **File Operations**
   - Upload (drag & drop or click)
   - Download
   - Delete (with confirmation)
   - Rename
   - Move/copy
   - Create folders

3. **File Preview**
   - Image preview with thumbnails
   - Text file editing
   - TOML config editing with syntax highlighting
   - JSON/YAML viewing

4. **Integration with Training**
   - Click image → Add to dataset
   - Click config → Load in training UI
   - Right-click → Quick actions

### Implementation

Using **elFinder** (already in your dependencies) or custom React components:

```typescript
// app/file-manager/page.tsx
'use client';

import { FileTree } from '@/components/FileTree';
import { FilePreview } from '@/components/FilePreview';
import { FileUpload } from '@/components/FileUpload';

export default function FileManagerPage() {
  const [selectedFile, setSelectedFile] = useState(null);

  return (
    <div className="grid grid-cols-12 h-screen">
      {/* Left sidebar: File tree */}
      <div className="col-span-3 border-r">
        <FileTree
          onSelect={setSelectedFile}
          rootPath="/workspace"
        />
      </div>

      {/* Main area: File preview/editor */}
      <div className="col-span-9">
        {selectedFile ? (
          <FilePreview file={selectedFile} />
        ) : (
          <FileUpload onUpload={handleUpload} />
        )}
      </div>
    </div>
  );
}
```

### Backend API

```python
# api/files.py
from fastapi import APIRouter, UploadFile
from pathlib import Path

router = APIRouter(prefix="/api/files")

@router.get("/list")
async def list_files(path: str = "/workspace"):
    """List files and directories"""
    p = Path(path)
    return {
        "files": [
            {
                "name": f.name,
                "path": str(f),
                "type": "dir" if f.is_dir() else "file",
                "size": f.stat().st_size if f.is_file() else 0,
                "modified": f.stat().st_mtime
            }
            for f in p.iterdir()
        ]
    }

@router.post("/upload")
async def upload_file(file: UploadFile, destination: str):
    """Upload file to specified path"""
    dest_path = Path(destination) / file.filename
    with dest_path.open("wb") as f:
        content = await file.read()
        f.write(content)
    return {"success": True, "path": str(dest_path)}

@router.delete("/{path:path}")
async def delete_file(path: str):
    """Delete file or directory"""
    p = Path(path)
    if p.is_file():
        p.unlink()
    elif p.is_dir():
        shutil.rmtree(p)
    return {"success": True}
```

## Deployment Scenarios

### 1. VastAI / RunPod (Remote GPU)

User rents GPU server with your template:
- Container starts automatically
- Services launch via `start_services.sh`
- VastAI exposes ports with URLs:
  - `https://ssh4.vast.ai:12345` → Next.js UI (main)
  - `https://ssh4.vast.ai:12346` → FastAPI docs
  - `https://ssh4.vast.ai:12347` → Jupyter Lab (optional)

### 2. Local Desktop

User clones repo and runs:
```bash
./start.sh
# Opens http://localhost:3000
```

### 3. Docker Compose (Future)

```yaml
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    volumes: ["./workspace:/workspace"]

  frontend:
    build: ./frontend
    ports: ["3000:3000"]
    depends_on: [backend]
```

## Jupyter Lab (Optional)

Jupyter Lab remains available for:
- **Power users** who want to experiment
- **Debugging** with interactive notebooks
- **Custom workflows** not covered by UI

Access: Port 8888 (automatically configured by VastAI base image)

**BUT:** Most users will never need to open it!

## Migration Path

### Phase 1: Basic Infrastructure ✅
- [x] VastAI provisioning script
- [x] Service startup script
- [x] Architecture documentation

### Phase 2: Backend API (Next)
- [ ] Create FastAPI app structure
- [ ] Wrap existing managers with API endpoints
- [ ] Add WebSocket for training logs
- [ ] File operations API

### Phase 3: Frontend Core
- [ ] Next.js app setup (TypeScript + Tailwind)
- [ ] File manager component
- [ ] Training configuration UI
- [ ] Progress monitoring UI

### Phase 4: Dataset Features
- [ ] Image upload with drag & drop
- [ ] Tag editor interface
- [ ] Dataset preview gallery
- [ ] Captioning tools

### Phase 5: Advanced Features
- [ ] Parameter calculator
- [ ] Model resizer
- [ ] HuggingFace integration
- [ ] Real-time training metrics

### Phase 6: Polish
- [ ] Responsive design
- [ ] Dark mode
- [ ] Keyboard shortcuts
- [ ] Documentation

## Tech Stack

### Frontend
- **Framework:** Next.js 14 (App Router)
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **UI Components:** shadcn/ui
- **File Upload:** react-dropzone
- **Code Editor:** Monaco Editor (VS Code engine)
- **Charts:** Recharts (for training metrics)

### Backend
- **Framework:** FastAPI
- **Language:** Python 3.10+
- **Async:** asyncio / aiofiles
- **WebSockets:** uvicorn
- **Existing Code:** All your managers stay as-is!

### Infrastructure
- **Container:** VastAI base image
- **Reverse Proxy:** Caddy (included in VastAI image)
- **Process Manager:** Supervisor (optional, for production)

## Benefits Over Jupyter

| Feature | Jupyter Notebooks | Next.js App |
|---------|------------------|-------------|
| **UI Design** | ❌ Limited styling | ✅ Full control |
| **File Manager** | ⚠️ Jupyter Lab | ✅ Built-in |
| **Memory** | ~300MB | ~100MB |
| **Mobile Support** | ❌ Poor | ✅ Responsive |
| **Real-time Updates** | ⚠️ Manual refresh | ✅ WebSockets |
| **Error Handling** | ❌ Cell-based | ✅ Global |
| **User Experience** | ⚠️ Developer-focused | ✅ User-friendly |
| **Deployment** | ⚠️ Complex | ✅ Simple |
| **Debugging** | ✅ Interactive | ⚠️ Logs-based |

**Verdict:** Next.js for main UI, keep Jupyter for debugging! 🎉

## Next Steps

Ready to start building? See:
1. `docs/BACKEND_API.md` - API endpoint specifications
2. `docs/FRONTEND_SETUP.md` - Next.js project structure
3. `docs/VASTAI_DEPLOYMENT.md` - Complete deployment guide
