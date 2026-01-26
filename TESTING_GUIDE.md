# Testing Guide - January 2026 Migration

**Current Status**: Node.js migration is 75% complete. Core features work, some rough edges expected.

## ✅ What's Ready to Test RIGHT NOW

### 1. WD14 Tagging (Node.js ONNX Runtime) - CRITICAL FIX
**Status**: Complete and testable on CPU (no GPU needed!)

**Test locally:**
```bash
cd frontend
npx ts-node test-tagging.ts path/to/any/image.jpg
```

**What to expect:**
- Loads WD14 model from cache (~1-2 seconds first time)
- Tags image in ~300ms on CPU
- Outputs list of detected tags with confidence scores

**This fixes the Windows asyncio issues!**

---

### 2. Config API Routes (Training Configuration)
**Status**: Complete - all 10 routes working

**Test with frontend:**
1. Start services: `cd frontend && npm run dev`
2. Navigate to training page
3. Create/load/save training configs
4. Generate TOML files for training

**Test with curl:**
```bash
# Get default config
curl http://localhost:3000/api/config/defaults

# List presets
curl http://localhost:3000/api/config/presets

# Validate config
curl -X POST http://localhost:3000/api/config/validate \
  -H "Content-Type: application/json" \
  -d '{"config": {...}}'
```

---

### 3. Previous Migrations (Phases 1-4)
**Status**: Already working and stable

**Working features:**
- ✅ File browser and operations
- ✅ Caption editing (add trigger, remove tags, replace text)
- ✅ Settings management (HF token, Civitai key)
- ✅ Job management (training, captioning jobs)

---

## ⚠️ Known Limitations (Still Using Python API)

These still use the Python FastAPI backend (port 8000):

### Dataset Operations
- **Upload files** - Works via Python API
- **Upload ZIP** - Works via Python API
- **Download from URL** - Works via Python API
- **List datasets** - Works via Python API

**Why**: File upload routes not yet migrated to Node.js

### Model Downloads
- **Download models/VAEs** - Works via Python API
- **Civitai integration** - Works via Python API

**Why**: Download orchestration not yet migrated

### ML Operations (Staying in Python)
- **Training** - Python subprocess (working)
- **BLIP Captioning** - Python subprocess (working)
- **GIT Captioning** - Python subprocess (working)

**Why**: These MUST stay Python (PyTorch/transformers)

---

## 🧪 Full Testing Workflow

### Local Testing (No GPU):
```bash
# 1. Start services
cd frontend
npm run dev

# Frontend: http://localhost:3000
# Node.js API: http://localhost:3000/api (automatic)

# 2. Test WD14 tagger (CPU)
npx ts-node test-tagging.ts ../dataset/test/image.jpg

# 3. Test config system via UI
# - Navigate to http://localhost:3000/training
# - Create training config
# - Save it
# - Check that TOML files are generated
```

### VastAI Testing (With GPU):
```bash
# 1. Deploy to VastAI
# 2. Access via portal URLs (ports 13000/18000)
# 3. Test full workflow:
#    - Upload dataset
#    - Run WD14 tagging (Node.js ONNX)
#    - Edit captions
#    - Configure training
#    - Start training
```

---

## 🐛 How to Report Issues

**GitHub Issues**: [Report Here](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues)
**Discord**: [Join Server](https://discord.gg/HhBSM9gBY)

**When reporting, please include:**
1. What you were trying to do
2. What happened (error message, screenshot)
3. Environment (local Windows/Linux, VastAI, etc.)
4. Browser console logs if relevant (F12 → Console tab)

---

## 📋 What Still Needs Work

**Optional features (Python API still works for these):**
- Dataset upload routes (Node.js version incomplete)
- Model download routes (not yet migrated)
- Civitai API routes (not yet migrated)
- Utilities routes (calculator, etc.)

**These are NOT BLOCKING** - the Python API handles them fine.

---

## 🎯 Priority for Next Testing

1. **WD14 Tagging** - Critical fix, test on real datasets
2. **Config System** - Test TOML generation for training
3. **Full Training Workflow** - Upload → Tag → Configure → Train

**Timeline**: 
- This week: Local testing
- Next week: VastAI GPU testing
- Following week: Bug fixes and polish

---

## 💡 Tips for Testing

**Development is part-time this month** - be patient with updates!

**If something breaks:**
- Check browser console (F12) for errors
- Try refreshing the page
- Clear browser cache if UI acts weird
- Report it on GitHub or Discord

**Node.js routes use relative URLs** - everything goes through `/api` automatically, so no config changes needed!

---

**Ready to test?** Start with the WD14 tagger test script - it's the quickest way to verify the migration worked! 🚀
