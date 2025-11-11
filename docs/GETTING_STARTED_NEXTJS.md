# Getting Started with Next.js UI

Welcome to the new Next.js-based interface! No more Jupyter widget race conditions! 🎉

## Quick Start (Local Development)

### 1. Start the Backend

```bash
# Install API dependencies
pip install -r requirements-api.txt

# Start FastAPI server
cd api
python main.py
```

Backend will run on http://localhost:8000

### 2. Start the Frontend

```bash
# Install Node dependencies
cd frontend
npm install

# Start development server
npm run dev
```

Frontend will run on http://localhost:3000

### 3. Access the UI

Open your browser to: **http://localhost:3000**

You'll see:
- 📁 File Manager
- 🖼️ Dataset Management
- ⚙️ Training Configuration
- 📊 Real-time Training Logs

## VastAI Deployment

### Option 1: Use Template (Easiest!)

1. Create template following `docs/VASTAI_TEMPLATE_GUIDE.md`
2. Rent GPU instance
3. Wait 5-10 minutes for setup
4. Click Port 3000 link → Start training!

### Option 2: Manual Setup

```bash
# Clone repo
git clone --recursive https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
cd Ktiseos-Nyx-Trainer

# Install dependencies
pip install -r requirements-api.txt
cd frontend && npm install && npm run build && cd ..

# Start services
./start_services.sh
```

## Features Tour

### File Manager (`/files`)

**Replaces Jupyter Lab!**

- ✅ Drag & drop file upload
- ✅ Browse directories
- ✅ Delete/rename files
- ✅ Create folders
- ✅ Download files
- ✅ No more widget bugs!

**Usage:**
1. Navigate through folders by clicking
2. Drag files from your computer to upload
3. Click file to preview/download
4. Right-side icons for actions

### Dataset Management (`/dataset`)

- Upload training images (batch upload!)
- Auto-tag with WD14 tagger
- Preview dataset gallery
- Organize into folders

**Usage:**
1. Create new dataset or select existing
2. Drag & drop images
3. Click "Auto-Tag" → Select model → Start
4. Review tags in file manager

### Training (`/training`)

- Configure all training parameters
- Start/stop training
- Real-time progress monitoring
- WebSocket log streaming (no polling!)

**Usage:**
1. Load config template or create new
2. Select dataset
3. Configure parameters (resolution, steps, LR, etc.)
4. Click "Start Training"
5. Watch logs stream in real-time!

### Configuration (`/configs`)

- Save/load training configs
- Template library
- TOML file editing

## Architecture

```
User Browser (Port 3000)
         ↓
   Next.js Frontend
   - React components
   - Tailwind CSS
   - TypeScript
         ↓ REST API / WebSocket
   FastAPI Backend (Port 8000)
   - Your Python managers
   - File operations
   - Training control
         ↓
   Kohya SD-Scripts
   - Actual training
   - On GPU
```

## Development Workflow

### Adding New Features

1. **Backend:** Add route in `api/routes/`
2. **Frontend:** Add API call in `lib/api.ts`
3. **UI:** Create component in `components/`
4. **Page:** Add page in `app/yourpage/`

### Testing

```bash
# Backend
cd api
pytest

# Frontend
cd frontend
npm run lint
npm run build  # Check for errors
```

### Hot Reload

Both backend and frontend support hot reload:
- Change Python code → FastAPI reloads
- Change React code → Next.js reloads instantly

## Common Tasks

### Upload Dataset

1. Go to File Manager
2. Navigate to `/workspace/datasets/`
3. Create new folder (e.g., "my_lora")
4. Drag & drop images into folder
5. Go to Dataset page → Auto-tag

### Start Training

1. Go to Training page
2. Load template (e.g., "sdxl_lora_default")
3. Set dataset path: `/workspace/datasets/my_lora`
4. Set output path: `/workspace/output/my_lora`
5. Adjust parameters
6. Click "Start Training"
7. Monitor logs in real-time!

### Download Trained Model

1. Training completes
2. Go to File Manager
3. Navigate to `/workspace/output/my_lora/`
4. Click download icon on `.safetensors` file

## Advantages Over Jupyter

| Feature | Jupyter Widgets | Next.js UI |
|---------|----------------|------------|
| **Race Conditions** | ❌ Common! | ✅ None! |
| **State Management** | ❌ Buggy | ✅ Proper React state |
| **File Upload** | ❌ Breaks often | ✅ Drag & drop, async |
| **Real-time Updates** | ⚠️ Polling | ✅ WebSockets |
| **UI Design** | ❌ Limited | ✅ Full control |
| **Mobile Support** | ❌ Poor | ✅ Responsive |
| **Debugging** | ❌ Difficult | ✅ Browser DevTools |
| **Performance** | ⚠️ Gets slow | ✅ Fast |

## Troubleshooting

### Backend won't start

```bash
# Check dependencies
pip list | grep fastapi

# Check port
lsof -i :8000  # Should be empty

# Run with debug
cd api
uvicorn main:app --reload --log-level debug
```

### Frontend won't start

```bash
# Clean install
rm -rf node_modules .next
npm install
npm run dev

# Check Node version
node --version  # Should be 18+
```

### Can't connect frontend to backend

1. Check backend is running: http://localhost:8000/docs
2. Check CORS settings in `api/main.py`
3. Check `.env.local` in frontend

### File uploads fail

1. Check disk space: `df -h`
2. Check permissions on `/workspace/`
3. Check file size limits in nginx (if using)

## Next Steps

1. **Explore the UI** - Click around, try features
2. **Upload a test dataset** - Small one first!
3. **Run a test training** - 100 steps to verify GPU works
4. **Check the logs** - WebSocket streaming is cool!
5. **Enjoy no race conditions!** 🎉

## Getting Help

- 📖 **Docs:** `docs/` folder
- 🐛 **Issues:** GitHub Issues
- 💬 **Discord:** [Your Discord]
- 🔧 **API Docs:** http://localhost:8000/docs (auto-generated!)

---

**Welcome to the future of LoRA training!** No more Jupyter widget pain! 🚀
