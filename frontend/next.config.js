/** @type {import('next').NextConfig} */
const nextConfig = {
  // API backend proxy
  // Routes /api/* requests to the backend server
  async rewrites() {
    // For VastAI/Docker: Backend is on localhost:8000
    // For local dev: Backend is on localhost:8000 or 127.0.0.1:8000
    const backendUrl = process.env.BACKEND_URL || 'http://127.0.0.1:8000';

    return [
      {
        source: '/api/:path*',
        destination: `${backendUrl}/api/:path*`,
      },
      // Note: WebSocket connections (/ws/*) connect directly to backend
      // Next.js rewrites don't support ws:// protocol
      // Frontend uses getWsUrl() to construct WebSocket URLs
    ];
  },
  // Note: Body size limits are handled by the FastAPI backend, not Next.js
  // See api/routes/dataset.py for upload configurations

  // Disable strict mode for development (optional)
  reactStrictMode: true,
  // Note: Standalone mode disabled for VastAI deployment (use normal build)
  // output: 'standalone',
  // Ignore ESLint errors during build (keep linting for dev)
  eslint: {
    ignoreDuringBuilds: true,
  },
  // Ignore TypeScript errors during build (keep type checking for dev)
  typescript: {
    ignoreBuildErrors: true,
  },
};

module.exports = nextConfig;
