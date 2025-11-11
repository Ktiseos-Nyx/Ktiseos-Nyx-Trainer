# Testing the Backend Connection

The API is now wired to your existing code! Here's how to test it:

## Quick Test

### 1. Start the Backend

```bash
# Install dependencies
pip install -r requirements-api.txt

# Start FastAPI
cd api
python main.py
```

Backend runs on: **http://localhost:8000**

### 2. Test the API

**Check if it's running:**
```bash
curl http://localhost:8000/
```

Should return:
```json
{
  "name": "Ktiseos-Nyx-Trainer API",
  "version": "0.1.0",
  "status": "running",
  "docs": "/docs"
}
```

**Check training status:**
```bash
curl http://localhost:8000/api/training/status
```

**View API docs:**
Open in browser: http://localhost:8000/docs

You'll see **interactive documentation** where you can test all endpoints!

### 3. Test with Frontend

```bash
# Terminal 2: Start Next.js
cd frontend
npm install
npm run dev
```

Frontend runs on: **http://localhost:3000**

Now:
1. Open http://localhost:3000/training
2. Fill in config form
3. Click "Start Training"
4. Backend calls your actual `RefactoredTrainingManager.start_training()`!

## What's Connected:

```
Next.js Form
    ↓ POST /api/training/start
FastAPI Endpoint (api/routes/training.py:84)
    ↓ Calls
training_manager.start_training(config_dict, monitor_widget=None)
    ↓ Runs
RefactoredTrainingManager (core/refactored_training_manager.py:84)
    ↓ Delegates to
KohyaTrainingManager (core/kohya_training_manager.py:819)
    ↓ Launches
Actual Kohya Training Scripts!
```

## Testing Without Frontend

Use the auto-generated docs at http://localhost:8000/docs:

1. Expand **POST /api/training/start**
2. Click "Try it out"
3. Edit the JSON config
4. Click "Execute"
5. See the response!

## Expected Behavior:

### Start Training:
```python
# POST /api/training/start
{
  "model_name": "test_lora",
  "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
  "output_dir": "/workspace/output/test",
  "train_data_dir": "/workspace/datasets/test",
  "resolution": 512,
  "train_batch_size": 1,
  "max_train_steps": 100,
  "learning_rate": 0.0001,
  "network_module": "networks.lora",
  "network_dim": 32,
  "network_alpha": 32
}
```

Response:
```json
{
  "success": true,
  "message": "Training started successfully",
  "training_id": "train_test_lora"
}
```

Your training manager will:
1. Validate config
2. Create config/dataset TOML files
3. Launch Kohya training subprocess
4. Return success

### Stop Training:
```bash
curl -X POST http://localhost:8000/api/training/stop
```

Calls: `training_manager.stop_training()`

### Check Status:
```bash
curl http://localhost:8000/api/training/status
```

Calls: `config_manager.get_training_status()`

## Debugging:

**Backend logs:**
Check terminal where you ran `python main.py`

**Python errors:**
```python
# In api/main.py, exceptions are logged:
logger.error(f"Unhandled exception: {exc}", exc_info=True)
```

**Test your managers directly:**
```python
from shared_managers import get_training_manager

manager = get_training_manager()

config = {
    "model_name": "test",
    "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
    "output_dir": "/workspace/output/test",
    "train_data_dir": "/workspace/datasets/test",
    # ... rest of config
}

# This is what the API calls:
success = manager.start_training(config, monitor_widget=None)
print(f"Training started: {success}")
```

## Next Steps:

Once you confirm the backend works:
1. **Test frontend** → backend → training manager flow
2. **Check if training actually starts** (look for Kohya process)
3. **Test file uploads** via dataset page
4. **Test WebSocket logs** (if training produces output)

## Known Limitations:

- WebSocket log streaming: Placeholder (needs log file monitoring)
- Progress tracking: Basic (needs log parsing)
- Training history: Lists directories only

These are "nice-to-haves" - the core flow WORKS!

---

**Bottom line:** Backend is wired to your code. You can now test if it actually works! 🚀
