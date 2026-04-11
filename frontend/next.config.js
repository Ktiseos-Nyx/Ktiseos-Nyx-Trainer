/** `@type` {import('next').NextConfig} */
const path = require('path');

const nextConfig = {
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
      fallback: [
        {
          source: '/api/:path*',
          destination: `${backendUrl}/api/:path*`,
        },
      ],
    };
  },

  reactStrictMode: true,
  eslint: { ignoreDuringBuilds: true },
  typescript: { ignoreBuildErrors: true },

};

module.exports = nextConfig;