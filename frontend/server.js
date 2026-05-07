/**
 * Custom Next.js Server with WebSocket Proxy Support
 *
 * This wraps Next.js with a custom HTTP server that properly proxies
 * WebSocket connections to the FastAPI backend on port 8000.
 *
 * Why: Cloudflare (and other reverse proxies) need proper HTTP upgrade
 * headers for WebSocket connections. Node.js handles this natively,
 * just like Jupyter uses Tornado for the same purpose.
 *
 * Usage:
 *   Development: npm run dev (still uses next dev)
 *   Production:  npm run start (uses this custom server)
 */

process.env.NODE_ENV = process.env.NODE_ENV || 'production';

const { createServer } = require('http');
const next = require('next');

// Guard: this project requires `node server.js` for WebSocket support.
// Running `next start` bypasses this server and breaks WebSocket proxying.
if (require.main === module && process.argv[1]?.includes('next') && !process.argv[1].endsWith('server.js')) {
  console.error('❌ This project requires `node server.js` for production. Do not run `next start`.');
  process.exit(1);
}
const { createProxyMiddleware } = require('http-proxy-middleware');
const { WebSocketServer } = require('ws');

// ========== Global Error Handlers ==========
// Prevent silent crashes on VastAI. Log the error and exit gracefully.

process.on('uncaughtException', (err) => {
  console.error('💀 Uncaught Exception:', err);
  // Give logs time to flush, then exit
  setTimeout(() => process.exit(1), 1000);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('💀 Unhandled Rejection at:', promise, 'reason:', reason);
  // Don't exit - unhandled rejections are often recoverable
});

// Environment setup
const dev = process.env.NODE_ENV !== 'production';
// In production (Docker/VastAI), always bind to 0.0.0.0 for container networking
// In dev, use 127.0.0.1 explicitly to avoid IPv6 resolution issues on Windows
// (localhost can resolve to ::1 on some Windows configs, breaking backend proxy)
// Don't use HOSTNAME env var (Docker sets it to container ID)
const hostname = dev ? '127.0.0.1' : '0.0.0.0';
const port = parseInt(process.env.PORT || '3000', 10);
const defaultBackendPort = dev ? '8000' : (process.env.BACKEND_PORT || '18000');
const backendUrl = process.env.BACKEND_URL || `http://127.0.0.1:${defaultBackendPort}`;

// WHATWG URL helper — replaces deprecated url.parse() (DEP0169)
// Next.js handle() expects { pathname, query } so we build that from URL.
const BASE_URL = 'http://localhost'; // only used for parsing relative URLs
function parseUrl(reqUrl) {
  const url = new URL(reqUrl, BASE_URL);
  // Build query object preserving repeated keys as arrays
  // Use null-prototype object to prevent prototype pollution from query keys
  const query = Object.create(null);
  for (const [key, value] of url.searchParams) {
    if (key in query) {
      query[key] = Array.isArray(query[key])
        ? [...query[key], value]
        : [query[key], value];
    } else {
      query[key] = value;
    }
  }
  return { pathname: url.pathname, query, search: url.search, url };
}

// Initialize Next.js app
const app = next({ dev, hostname, port });
const handle = app.getRequestHandler();

console.log('🚀 Starting Ktiseos-Nyx-Trainer Custom Server...');
console.log(`   Environment: ${dev ? 'development' : 'production'}`);
console.log(`   Frontend: ${hostname}:${port}`);
console.log(`   Backend: ${backendUrl}`);

app.prepare().then(() => {
  // Create WebSocket + API proxy to FastAPI backend
  const apiAndWsProxy = createProxyMiddleware({
    target: backendUrl,
    changeOrigin: true,
    // ws: true is intentionally omitted — upgrade events are handled manually
    // in server.on('upgrade') below to avoid double-handling
    logLevel: dev ? 'debug' : 'warn',

    // Preserve original path (don't strip /api or /ws prefix)
    pathRewrite: (path, req) => {
      return path;
    },

    // Handle proxy errors gracefully
    onError: (err, req, res) => {
      console.error('❌ Proxy error:', err.message);
      if (res.writeHead) {
        res.writeHead(502, {
          'Content-Type': 'application/json',
        });
        res.end(JSON.stringify({
          error: 'Backend service unavailable',
          message: 'FastAPI backend is not responding. Is it running on port 8000?',
        }));
      }
    },

    // Log successful proxying (only in dev)
    onProxyReq: (proxyReq, req, res) => {
      if (dev && req.url.startsWith('/ws')) {
        console.log(`🔌 Proxying WebSocket: ${req.url}`);
      }
    },
  });

  // Request logging for production — Next.js dev mode logs automatically,
  // but the custom production server is silent without this.
  const logRequest = (req, res) => {
    const { pathname } = parseUrl(req.url);
    // Skip static assets, health checks, and favicon to reduce noise
    const isSkipped = pathname.startsWith('/_next/')
      || pathname.startsWith('/favicon')
      || pathname === '/health'
      || pathname === '/api/health';
    const start = Date.now();

    // Hook into response finish to log with status code
    res.on('finish', () => {
      if (isSkipped) return;
      const duration = Date.now() - start;
      const slow = duration > 5000 ? ' ⚠️ slow' : '';
      console.log(`${req.method} ${pathname} → ${res.statusCode} (${duration}ms${slow})`);
    });
  };

  // Create HTTP server
  const server = createServer(async (req, res) => {
    try {
      const parsedUrl = parseUrl(req.url);
      const { pathname } = parsedUrl;

      // Log requests in production only (dev mode has its own verbose logging)
      if (!dev) logRequest(req, res);

      // Handle Node.js API routes (new migration)
      const nodeApiPrefixes = ['/api/jobs', '/api/files', '/api/captions', '/api/settings', '/api/dataset', '/api/config', '/api/civitai', '/api/utilities', '/api/debug'];
      const nodeApiExact = ['/api/models/popular', '/api/models/list'];
      const isNodeApi = nodeApiPrefixes.some(prefix => pathname.startsWith(prefix)) || nodeApiExact.includes(pathname);

      if (isNodeApi) {
        try {
          return await handle(req, res, parsedUrl);
        } catch (e) {
          if (e.message?.includes('parsedUrl') || e.message?.includes('argument')) {
            console.warn('⚠️ Next.js handle() signature changed; retrying without parsedUrl');
            return await handle(req, res);
          }
          throw e;
        }
      }

      // Proxy other /api requests to FastAPI (backward compatibility)
      if (pathname.startsWith('/api')) {
        return apiAndWsProxy(req, res);
      }

      // /ws paths are WebSocket-only — reject plain HTTP requests
      // Actual upgrade handling is in server.on('upgrade') below
      if (pathname.startsWith('/ws')) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'WebSocket upgrade required' }));
        return;
      }

      // Handle all other requests with Next.js
      try {
        await handle(req, res, parsedUrl);
      } catch (e) {
        if (e.message?.includes('parsedUrl') || e.message?.includes('argument')) {
          console.warn('⚠️ Next.js handle() signature changed; retrying without parsedUrl');
          await handle(req, res);
        } else {
          throw e;
        }
      }
    } catch (err) {
      console.error('Error handling request:', err);
      res.statusCode = 500;
      res.end('Internal server error');
    }
  });

  // Create native WebSocket server for Node.js job logs
  const wss = new WebSocketServer({ noServer: true });

  // Handle WebSocket connections
  wss.on('connection', (ws, req) => {
    const { pathname } = parseUrl(req.url);
    console.log(`✅ WebSocket connected: ${pathname}`);

    // Extract job ID from path: /ws/jobs/{jobId}/logs
    const match = pathname.match(/^\/ws\/jobs\/([^/]+)\/logs$/);
    if (!match) {
      ws.close(4000, 'Invalid WebSocket path');
      return;
    }

    const jobId = match[1];

    // Use shared event bus (plain JS, works across server.js and Next.js API routes)
    const { jobEvents, jobsMap } = require('./lib/node-services/job-events');
    const job = jobsMap.get(jobId);

    if (!job) {
      ws.send(JSON.stringify({ error: `Job ${jobId} not found` }));
      ws.close(4004, 'Job not found');
      return;
    }

    // Send existing logs
    job.logs.forEach((log) => {
      ws.send(JSON.stringify({ type: 'log', log }));
    });

    // Send current status
    ws.send(JSON.stringify({
      type: 'status',
      status: job.status,
      progress: job.progress,
    }));

    // Listen for new logs
    const logListener = (jId, log) => {
      if (jId === jobId && ws.readyState === ws.OPEN) {
        ws.send(JSON.stringify({ type: 'log', log }));
      }
    };

    const statusListener = (jId, status) => {
      if (jId === jobId && ws.readyState === ws.OPEN) {
        ws.send(JSON.stringify({ type: 'status', status }));
      }
    };

    const progressListener = (jId, progress) => {
      if (jId === jobId && ws.readyState === ws.OPEN) {
        ws.send(JSON.stringify({ type: 'progress', progress }));
      }
    };

    jobEvents.on('log', logListener);
    jobEvents.on('status', statusListener);
    jobEvents.on('progress', progressListener);

    // Handle client disconnect
    ws.on('close', () => {
      console.log(`🔌 WebSocket disconnected: ${pathname}`);
      jobEvents.removeListener('log', logListener);
      jobEvents.removeListener('status', statusListener);
      jobEvents.removeListener('progress', progressListener);
    });

    // Handle ping/pong for connection health
    ws.on('ping', () => ws.pong());
  });

  // Handle WebSocket upgrade requests
  // This is the critical part for Cloudflare compatibility
  server.on('upgrade', (req, socket, head) => {
    let pathname;
    try {
      ({ pathname } = parseUrl(req.url));
    } catch (err) {
      console.error('❌ Malformed WebSocket URL:', req.url, err.message);
      socket.destroy();
      return;
    }

    // Native Node.js WebSocket for /ws/jobs/* paths
    if (pathname.startsWith('/ws/jobs/')) {
      console.log(`⬆️  WebSocket upgrade (Node.js): ${pathname}`);
      wss.handleUpgrade(req, socket, head, (ws) => {
        wss.emit('connection', ws, req);
      });
    }
    // Proxy WebSocket upgrades for /ws/api/* paths to FastAPI
    else if (pathname.startsWith('/ws/api/')) {
      console.log(`⬆️  WebSocket upgrade (FastAPI proxy): ${pathname}`);
      apiAndWsProxy.upgrade(req, socket, head);
    }
    else {
      // Reject other upgrade attempts with a proper HTTP response
      socket.write('HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{"error":"WebSocket upgrade rejected: unsupported path"}');
      socket.destroy();
    }
  });

  // Handle server errors
  server.on('error', (err) => {
    console.error('❌ Server error:', err);
    process.exit(1);
  });

  // Start listening
  server.listen(port, hostname, (err) => {
    if (err) throw err;
    console.log('');
    console.log('========================================');
    console.log('✅ Ktiseos-Nyx-Trainer Ready!');
    console.log('========================================');
    console.log(`🌐 Frontend:  http://${hostname === '0.0.0.0' ? 'localhost' : hostname}:${port}`);
    console.log(`🐍 Backend:   ${backendUrl}`);
    console.log(`🔌 WebSocket (Node.js):  /ws/jobs/{id}/logs`);
    console.log(`🔌 WebSocket (FastAPI):  /ws/api/* (proxied)`);
    console.log('========================================');
    console.log('');
  });

  // Graceful shutdown
  const shutdown = () => {
    console.log('\n🛑 Shutting down server...');
    server.close(() => {
      console.log('✅ Server closed');
      process.exit(0);
    });
  };

  process.on('SIGTERM', shutdown);
  process.on('SIGINT', shutdown);
});
