# TODO: Async Model Downloads with Cancellation

## Current Problem
Model downloads are synchronous and block the FastAPI worker. This means:
- ❌ No way to cancel downloads mid-flight
- ❌ No real-time progress updates
- ❌ UI gets stuck in "downloading..." state
- ❌ Can't start new downloads while one is running

## Proper Solution (Future Implementation)

### Backend Changes
1. **Make downloads async background jobs**
   - Convert `model_service.py` to use `asyncio.create_subprocess_exec()` instead of `subprocess.run()`
   - Register download processes with `job_manager` (like training/tagging)
   - Return job_id immediately, run download in background

2. **Add download job tracking**
   - Track download progress (bytes downloaded / total bytes)
   - Parse aria2c/wget output for progress updates
   - Expose via WebSocket like training logs

3. **Implement proper cancellation**
   - Store subprocess reference in job_manager
   - Cancel endpoint kills the tracked subprocess
   - Clean up partial downloads

### Frontend Changes
1. **Add real-time progress**
   - WebSocket connection to `/ws/jobs/{job_id}/logs`
   - Progress bar showing % complete
   - Download speed display

2. **Add cancel button**
   - Always visible (accessibility!)
   - Calls `/api/models/cancel/{job_id}`
   - Shows confirmation when cancelled

3. **Better UX**
   - List of active downloads
   - Queue multiple downloads
   - Pause/resume support (aria2c supports this!)

## Reference Implementation
See how training/tagging work:
- `services/training_service.py` - Async subprocess with job tracking
- `services/tagging_service.py` - Job manager integration
- `frontend/app/dataset/auto-tag/page.tsx` - Real-time progress + cancel button

## Complexity Estimate
- **Backend:** ~4-6 hours (refactor download flow, job integration)
- **Frontend:** ~2-3 hours (progress UI, WebSocket integration)
- **Testing:** ~2 hours (test all download methods, cancellation)

**Total:** ~8-11 hours for proper implementation

## Quick Fix Applied (2025-11-30) ✅

**What was added:**
1. **Prominent download status indicator** on Models page (`/models/page.tsx`)
   - Large animated card with cyan border when download is active
   - Shows download type (model/VAE) and source (HuggingFace/Civitai)
   - Warning message about not closing the page

2. **Enhanced success/error notifications**
   - Success: Green gradient card with icon showing filename and size
   - Error: Red bordered card with clear error message

3. **Active downloads indicator** on Civitai Browse page (`/models/browse/page.tsx`)
   - Shows count of active downloads
   - Appears when any download is in progress

**Result:** Users now have clear visual feedback that downloads are happening

## For Now
- ✅ Downloads work and are fast (aria2c multi-connection)
- ✅ Filename preservation fixed (Civitai downloads save correctly)
- ✅ Visual feedback added - users can see when downloads are active
- ⚠️ Still synchronous (blocks backend) - proper async implementation planned for next couple weeks
