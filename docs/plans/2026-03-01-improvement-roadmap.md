# Improvement Roadmap

**Date**: 2026-03-01
**Context**: Post-tagger restoration audit of codebase health, installer paths, memory management, and future optimization opportunities.

---

## Tier 1: Reliability -- COMPLETED 2026-03-01

### 1A. Fix `getPythonPath()` and venv detection
**Risk: Medium | Files: `frontend/lib/node-services/job-manager.ts`, installer scripts**

`getPythonPath()` is a complete stub that always returns `'python'`. It builds venv path candidates but never checks them. Worse, it looks for `venv/` but `install.sh` creates `.venv/`. On VastAI, the Docker image venv is at `/venv/main/`.

The current setup works by accident:
- VastAI: supervisor activates `/venv/main/` before starting Node, so `python` on PATH is correct
- Windows: packages installed system-wide, so system `python` has the right deps

**Fix approach:**
1. Check for venv at multiple known locations: `.venv/`, `venv/`, `/venv/main/` (VastAI)
2. Fall back to `python3` then `python` on PATH
3. Cache the result (like `detectBackend()` in the orchestrator)
4. Optionally verify the found Python has required packages (`python -c "import onnxruntime"`)

### 1B. Add global error handlers to `server.js`
**Risk: Medium | Files: `frontend/server.js`**

No `process.on('uncaughtException')` or `process.on('unhandledRejection')` handlers exist. An unhandled rejection silently kills the server on VastAI with no trace.

**Fix:** Add handlers that log the error and optionally attempt graceful shutdown.

### 1C. Fix missing `installer_remote.py` references
**Risk: Low-Medium | Files: `start_services_vastai.sh`, `vastai_setup.sh`**

Both scripts reference `installer_remote.py` which doesn't exist in the repo. `start_services_vastai.sh` has a bug where both if/else branches try to call it. Currently rescued by `vastai_setup.sh` doing the real setup earlier, but the dead references should be cleaned up.

### 1D. Fix hardcoded SDXL training script path
**Risk: Medium | Files: `frontend/lib/node-services/job-manager.ts`**

`createTrainingJob()` hardcodes `sdxl_train_network.py`. SD1.5 needs `train_network.py`, Flux needs `flux_train_network.py`, SD3 needs `sd3_train_network.py`. The training config should specify which script to use based on the selected model architecture.

---

## Tier 2: Long-running Instance Health & Quick Wins

### 2A. npm Dead Weight Removal
**Risk: Low | Files: `frontend/package.json`**

Several packages are redundant or duplicated:
- **`axios`** (~13KB gzipped) — unused, all API calls use native `fetch` via Next.js
- **`framer-motion` AND `motion`** — same library (motion is framer-motion's rename). Shipping it twice. Pick one.
- **`gsap` + `@gsap/react`** (~60KB) — only used for rolling number counter and GitHub star counter on homepage. Could replace with CSS or motion.
- **`styled-jsx`** — Next.js includes this by default. Explicit dep is redundant.

### 2B. Add Toast Notifications (`sonner`)
**Risk: Low | Files: `frontend/app/layout.tsx`, various pages**

No toast/notification system exists. Errors, success messages, and warnings all go to `console.log` or disappear silently. `sonner` is ~3KB, accessible, themeable, and shadcn already has a component wrapper for it. Drop-in addition.

### 2C. Error log surfacing in UI
**Files: New `/system` or `/dashboard` page, `server.js`**

Currently all server errors go to `console.error()` only. No way to see them from the browser. On VastAI this means SSH-ing in to debug.

**Approach:**
- Server-level ring buffer for recent errors (last N entries)
- Expose via REST endpoint (`/api/debug/errors`) and optionally WebSocket
- Simple UI page showing failed jobs (`jobManager.getJobsByStatus('failed')`), recent errors, and memory stats (reuse `/api/debug/memory`)
- Gradio-style console panel as stretch goal

### 2D. ONNX idle session release
**Files: `frontend/lib/node-services/tagging-service.ts`**

The Node.js ONNX fallback holds 300-500MB in RAM after a model is loaded, indefinitely. Should auto-release after N minutes of inactivity.

**Approach:** Add a `lastUsed` timestamp, check on a timer, call `dispose()` if idle too long.

### 2E. Blob URL stale closure fix
**Files: `frontend/components/DatasetUploader.tsx`**

The unmount cleanup captures `files` at mount time (empty dep array). Files added later aren't revoked on unmount. Use a ref to always see latest file list.

---

## Tier 3: Quality of Life

### 3A. `next/image` for dataset previews
Images come from API routes and blob URLs, which limits `next/image` optimization benefits. Worth testing on one component first. Blob URLs need `unoptimized` prop.

### 3B. AbortSignal for subprocess cancellation
Kill Python processes when WebSocket client disconnects, preventing orphaned training jobs.

### 3C. Ring buffer for global server logs
Stream server-level logs (not just per-job) to the UI. Would power the Tier 2A error log page.

---

## Tier 4: Frontend Performance & Accessibility

Goal: Keep the app lightweight and accessible - friendly to browsers that aren't Chrome, won't eat RAM on modest machines, and works well with assistive technology.

### 4A. Bundle size audit
**Current state:** ~103KB first-load shared JS (good for Next.js). But individual pages vary widely - dataset tags page is 190KB, training page is 117KB.

**Opportunities to investigate:**
- `@tanstack/react-query` or `swr` for data fetching with built-in caching, deduplication, and stale-while-revalidate (reduces redundant API calls)
- `react-virtual` / `@tanstack/virtual` for virtualizing large tag lists and image galleries (dataset with 500+ images shouldn't render 500 DOM nodes)
- Tree-shaking audit of Radix UI imports (shadcn components can pull in more than expected)

### 4B. Reduce runtime memory footprint
**Opportunities:**
- `comlink` for offloading heavy work to Web Workers (e.g., image preview generation, tag parsing) without blocking the main thread
- `idb-keyval` or similar for moving large state (like full tag lists) out of React state and into IndexedDB
- Virtualized lists for any component showing 50+ items

### 4C. Accessibility improvements
**Current state:** shadcn/ui provides good baseline a11y, but custom components and pages may have gaps.

**Investigate:**
- `eslint-plugin-jsx-a11y` - lint for accessibility issues at build time (may already be partially configured via Next.js defaults)
- Keyboard navigation audit on key workflows (dataset upload, tag editing, training config)
- Screen reader testing on training monitor (live-updating content needs proper aria-live regions)
- Color contrast audit on the effects components (neon text, glitch effects, gradient borders)

### 4D. Performance monitoring
**Opportunities:**
- `web-vitals` (already included by Next.js, just needs to be wired to reporting)
- Lighthouse CI in GitHub Actions for regression detection
- The `/api/debug/memory` endpoint we added today as a starting point for server-side monitoring

### 4E. Efficient image handling
**Current state:** Images loaded as raw `<img>` tags with `loading="lazy"`.

**Investigate:**
- `blurhash` or `thumbhash` for placeholder previews while images load (much better UX than blank space)
- Canvas-based thumbnail generation client-side instead of loading full-res images for gallery views
- `Intersection Observer` for lazy-loading that's more reliable than native `loading="lazy"` across browsers

---

### 4F. Command Palette (`cmdk`)
Power-user navigation: Cmd+K to search pages, datasets, run actions. shadcn already has a component wrapper. Small package, big UX win for keyboard-heavy workflows.

### 4G. Fully adopt TanStack Query
`@tanstack/react-query` is installed but likely underused. Moving all API calls to use it gives: automatic caching, request deduplication, stale-while-revalidate, loading/error states, and retry logic — for free. Reduces custom `useEffect` + `useState` fetch patterns across the app.

---

## Future Considerations (Not YAGNI, Just Not Now)

- **Electron/Tauri desktop edition** — Reasonable long-term goal once the web app is stable and battle-tested. Tauri would be lighter than Electron. Not imperative until the core product works reliably.
- **Database/connection pooling** — No DB needed, don't add one for logging
- **Server-side caching layers** — Single-user app, not high-traffic
- **Complex process managers** — Supervisor already handles this on VastAI
- **Turbopack in production** — Dev-only, webpack is correct for builds
- **socket.io** — Native `ws` WebSocket works fine, no need for the abstraction layer
