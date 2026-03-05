# Development Environments

Technical documentation on development vs. production mode resource usage and platform-specific considerations.

## Resource Usage Comparison

### Development Mode (`npm run dev`)

**Memory Usage:**
- **Windows**: 1.5-2GB+ (Node.js process)
- **macOS/Linux**: 500MB-1GB (Node.js process)

**What's Running:**
- Next.js dev server with Turbopack bundler (Rust-based)
- TypeScript compilation daemon
- File system watchers (entire project tree)
- Hot Module Replacement (HMR) state
- Source map generation
- React Fast Refresh

**Purpose:** Active development with instant hot reload on file changes.

### Production Mode (`npm run build && npm start`)

**Memory Usage:**
- **All Platforms**: 200-300MB (Node.js process)

**What's Running:**
- Pre-built static file server
- SSR rendering for dynamic routes
- No watchers, no compilation

**Purpose:** Testing production builds, low-resource environments, or when not actively coding.

## Platform-Specific Behavior

### Windows

**File System Watchers:**
- Uses polling-based watchers (less efficient than Unix systems)
- Windows Defender/antimalware actively scans all file operations
- Higher memory fragmentation in Node.js runtime

**Memory Impact:** 2-3x higher RAM usage compared to Unix-like systems for development mode.

**Mitigation:**
- Turbopack (enabled by default) uses ~50% less RAM than Webpack
- Exclude `node_modules` and `.next` from antivirus real-time scanning
- Use production mode when testing features (not actively coding)

### macOS/Linux

**File System Watchers:**
- Native efficient APIs (`FSEvents` on macOS, `inotify` on Linux)
- Better memory management and garbage collection
- Lower overhead from system monitoring

**Memory Impact:** Standard Node.js development overhead (~500MB-1GB).

## Low-RAM Development Workflows

### Option 1: Production Testing

When RAM is constrained and you're testing features (not actively editing code):

```bash
cd frontend
npm run build    # ~30-60 seconds, compiles optimized bundle
npm start        # Runs production server at ~200-300MB RAM
```

**Trade-offs:**
- No hot reload (must rebuild to see changes)
- Optimized bundle (minified, tree-shaken)
- Lower resource usage

### Option 2: Selective Development

Keep dev server running only when actively coding:

```bash
# When coding:
npm run dev

# When testing/stepping away:
Ctrl+C  # Stop dev server

# Resume when needed:
npm run dev  # Restarts in ~10-15 seconds
```

### Option 3: Build Tuning (Advanced)

Limit Node.js heap size (forces more aggressive garbage collection):

```bash
NODE_OPTIONS='--max-old-space-size=512' npm run dev
```

**Warning:** May crash if 512MB is insufficient. Adjust as needed.

## Backend (FastAPI) Resource Usage

The Python backend uses minimal resources:

- **Memory**: ~100-200MB (idle)
- **Platform**: No significant Windows vs. Unix difference

The frontend (Next.js) is the primary RAM consumer in development.

## Recommended Configurations

### Windows Development (Low RAM)

```bash
# Terminal 1: Backend (minimal RAM)
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend production mode (200-300MB)
cd frontend
npm run build && npm start
```

**Total RAM:** ~300-500MB

### Windows Development (16GB+ RAM)

```bash
# Terminal 1: Backend
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend dev mode with Turbopack (1-1.5GB)
cd frontend
npm run dev
```

**Total RAM:** ~1.2-1.7GB

### macOS/Linux Development

```bash
# Terminal 1: Backend
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend dev mode (500MB-1GB)
cd frontend
npm run dev
```

**Total RAM:** ~600MB-1.2GB

## Troubleshooting

### "JavaScript heap out of memory" Error

Increase Node.js heap size:

```bash
NODE_OPTIONS='--max-old-space-size=2048' npm run dev
```

### Dev Server Slow on Windows

1. **Check antivirus exclusions** - Exclude project directory from real-time scanning
2. **Verify Turbopack is enabled** - Should see "(Turbopack)" in startup message
3. **Close unnecessary programs** - Dev server needs available RAM for file watchers

### Frequent Hot Reload Failures

Lower file watcher count by excluding large directories:

```js
// next.config.js
module.exports = {
  webpack: (config) => {
    config.watchOptions = {
      ignored: ['**/node_modules', '**/.git', '**/datasets']
    };
    return config;
  }
};
```

## Technical Details

### Why Windows Uses More RAM

1. **Polling-based file watchers** instead of event-driven (FSEvents/inotify)
2. **NTFS file system overhead** vs. ext4/APFS
3. **Windows process isolation model** vs. Unix shared memory
4. **Antivirus real-time scanning** adds overhead to every file operation
5. **Different V8 memory allocation patterns** on Windows vs. Unix

### Turbopack vs. Webpack

- **Webpack**: JavaScript-based bundler (~2GB RAM in dev mode)
- **Turbopack**: Rust-based bundler (~800MB-1.2GB RAM in dev mode)
- **Improvement**: 40-50% reduction in memory usage

Turbopack is enabled by default in this project (`--turbo` flag in `package.json`).
