/** `@type` {import('next').NextConfig} */
const path = require('path');

const nextConfig = {
  // Empty turbopack config confirms we know Turbopack is the default in Next 16.
  // Webpack config below is kept for `npm run dev:webpack` fallback only.
  turbopack: {},

  // Webpack-only: pin React to a single copy to prevent duplicate-runtime crashes
  // on Windows where NTFS path-casing can trick webpack into bundling react twice.
  // Turbopack (default in Next 16) handles this natively — no alias needed there.
  webpack: (config) => {
    config.resolve.alias = {
      ...(config.resolve.alias || {}),
      react: path.resolve(__dirname, 'node_modules/react'),
      'react-dom': path.resolve(__dirname, 'node_modules/react-dom'),
      'react/jsx-runtime': path.resolve(__dirname, 'node_modules/react/jsx-runtime'),
      'react/jsx-dev-runtime': path.resolve(__dirname, 'node_modules/react/jsx-dev-runtime'),
    };
    return config;
  },

  // ✅ CVE-2026-27980: Cap image optimization disk cache (added in Next.js 15.5.14)
  images: {
    maximumDiskCacheSize: 500 * 1024 * 1024, // 500 MB
  },

  // ✅ FIX: Allow massive uploads (2GB limit)
  experimental: {
    serverActions: {
      bodySizeLimit: '2gb',
    },
  },

  // ✅ FIX: Don't bundle native Node.js addons
  serverExternalPackages: ['onnxruntime-node', 'sharp'],

  // 🚀 PERFORMANCE
  compress: true,
  productionBrowserSourceMaps: false,

  // Fix workspace root warning (monorepo detection)
  outputFileTracingRoot: path.join(__dirname, '../'),

  // API backend proxy - FALLBACK only (custom server.js handles routing first)
  async rewrites() {
    const backendPort = process.env.BACKEND_PORT || '8000';
    const backendUrl = process.env.BACKEND_URL || `http://127.0.0.1:${backendPort}`;
    return {
      // FastAPI owns crop/convert, but they live under /api/dataset, which also
      // has a Next [name] dynamic route. A `fallback` rewrite loses to that
      // dynamic route (→ 405 on POST); `beforeFiles` runs before filesystem
      // routes, so these reach FastAPI. Mirrors the server.js carve-out for prod.
      beforeFiles: [
        { source: '/api/dataset/crop', destination: `${backendUrl}/api/dataset/crop` },
        { source: '/api/dataset/crop/:path*', destination: `${backendUrl}/api/dataset/crop/:path*` },
        { source: '/api/dataset/convert', destination: `${backendUrl}/api/dataset/convert` },
        { source: '/api/dataset/convert/:path*', destination: `${backendUrl}/api/dataset/convert/:path*` },
      ],
      fallback: [
        {
          source: '/api/:path*',
          destination: `${backendUrl}/api/:path*`,
        },
      ],
    };
  },

  reactStrictMode: true,
  typescript: { ignoreBuildErrors: true },

};

module.exports = nextConfig;
