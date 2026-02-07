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

const { createServer } = require('http');
const { parse } = require('url');
const next = require('next');
const { createProxyMiddleware } = require('http-proxy-middleware');
const { WebSocketServer } = require('ws');

// Environment setup
const dev = process.env.NODE_ENV !== 'production';
// In production (Docker/VastAI), always bind to 0.0.0.0 for container networking
// In dev, use localhost. Don't use HOSTNAME env var (Docker sets it to container ID)
const hostname = dev ? 'localhost' : '0.0.0.0';
const port = parseInt(process.env.PORT || '3000', 10);
const backendUrl = process.env.BACKEND_URL || 'http://127.0.0.1:8000';

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
    ws: true, // Enable WebSocket proxying
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

  // Create HTTP server
  const server = createServer(async (req, res) => {
    try {
      const parsedUrl = parse(req.url, true);
      const { pathname } = parsedUrl;

      // Handle Node.js API routes (new migration)
      const nodeApiPrefixes = ['/api/jobs', '/api/files', '/api/captions', '/api/settings'];
      const isNodeApi = nodeApiPrefixes.some(prefix => pathname.startsWith(prefix));

      if (isNodeApi) {
        // Let Next.js handle these routes
        return await handle(req, res, parsedUrl);
      }

      // Proxy other /api requests to FastAPI (backward compatibility)
      if (pathname.startsWith('/api')) {
        return apiAndWsProxy(req, res);
      }

      // Proxy WebSocket endpoint requests to FastAPI
      // Note: Actual upgrade happens in server.on('upgrade')
      if (pathname.startsWith('/ws')) {
        return apiAndWsProxy(req, res);
      }

      // Handle all other requests with Next.js
      await handle(req, res, parsedUrl);
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
    const { pathname } = parse(req.url, true);
    console.log(`✅ WebSocket connected: ${pathname}`);

    // Extract job ID from path: /ws/jobs/{jobId}/logs
    const match = pathname.match(/^\/ws\/jobs\/([^/]+)\/logs$/);
    if (!match) {
      ws.close(4000, 'Invalid WebSocket path');
      return;
    }

    const jobId = match[1];

    // Import job manager (lazy load to avoid circular deps)
    const { jobManager } = require('./lib/node-services/job-manager');
    const job = jobManager.getJob(jobId);

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

    jobManager.events.on('log', logListener);
    jobManager.events.on('status', statusListener);
    jobManager.events.on('progress', progressListener);

    // Handle client disconnect
    ws.on('close', () => {
      console.log(`🔌 WebSocket disconnected: ${pathname}`);
      jobManager.events.removeListener('log', logListener);
      jobManager.events.removeListener('status', statusListener);
      jobManager.events.removeListener('progress', progressListener);
    });

    // Handle ping/pong for connection health
    ws.on('ping', () => ws.pong());
  });

  // Handle WebSocket upgrade requests
  // This is the critical part for Cloudflare compatibility
  server.on('upgrade', (req, socket, head) => {
    const { pathname } = parse(req.url, true);

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
      // Reject other upgrade attempts
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
