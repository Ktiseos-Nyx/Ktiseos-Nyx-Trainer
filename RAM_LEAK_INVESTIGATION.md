# Node.js RAM Leak Investigation

## Issue Discovered: 2026-01-03

### Symptoms
- **Node.js server-side** RAM usage increases dramatically during navigation
- **NOT** a Next.js frontend/client-side issue
- RAM increases when navigating from landing page to dashboard
- Behavior persists even with minimal user interaction

### Environment
- Windows 10
- Next.js 15 with React 19
- Development mode (need to verify if production mode has same issue)

### Suspected Causes
- [ ] Server-side rendering memory accumulation
- [ ] API route handlers not releasing resources
- [ ] WebSocket connections not being cleaned up
- [ ] Cache/state not being garbage collected
- [ ] React Server Components memory retention

### Reproduction Steps
1. Start services with `start_services_local.bat`
2. Navigate to landing page (http://localhost:3000)
3. Monitor Node.js RAM usage
4. Navigate to dashboard
5. Observe RAM spike

### Notes
- Accidentally spawned multiple Node instances during investigation (Claude ran dev + production simultaneously)
- Need to test in production mode vs dev mode
- Need to profile with Node.js memory tools

### Additional Finding: Orphaned Python Subprocess (2026-01-03)
**Symptom:** After closing PowerShell, a Python subprocess remained running in the background

**Root Cause:** `start_services_local.bat` uses `start /MIN` to launch services in detached windows
- Line 44: `start "Ktiseos Backend" /MIN python -m uvicorn ...`
- Line 67: `start "Ktiseos Frontend" /MIN cmd /c ...`
- These processes don't terminate when parent PowerShell window closes

**Impact:**
- Orphaned processes continue consuming RAM/CPU
- Conflicting processes when restarting services
- Resource leaks accumulate across dev sessions

**Related to RAM Leak:** Both issues compound - Node.js leaks memory AND orphaned processes prevent cleanup

### Next Steps
- [ ] Test in production mode to isolate dev-mode-specific issues
- [ ] Add Node.js `--inspect` flag and use Chrome DevTools memory profiler
- [ ] Check for unclosed connections in API routes
- [ ] Review WebSocket handler cleanup in `services/websocket.py` integration
- [ ] Check Next.js 15 known issues for server memory leaks
- [ ] **PRIORITY:** Create proper shutdown script for Windows that kills orphaned processes
- [ ] Consider using job objects or process groups for proper process lifecycle management
- [ ] Add process monitoring/cleanup to startup script

---

**Investigation Status:** Active
**Priority:** High (affects development experience and production viability)
**Severity Increase:** Orphaned processes + memory leak = critical development blocker
