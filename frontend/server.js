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

// Environment setup
const dev = process.env.NODE_ENV !== 'production';
const hostname = process.env.HOSTNAME || '0.0.0.0';
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

      // Proxy API requests to FastAPI
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

  // Handle WebSocket upgrade requests
  // This is the critical part for Cloudflare compatibility
  server.on('upgrade', (req, socket, head) => {
    const { pathname } = parse(req.url, true);

    // Only proxy WebSocket upgrades for /ws/* paths
    if (pathname.startsWith('/ws')) {
      console.log(`⬆️  WebSocket upgrade: ${pathname}`);
      apiAndWsProxy.upgrade(req, socket, head);
    } else {
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
    console.log(`🔌 WebSocket: Proxy enabled for /ws/* endpoints`);
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
