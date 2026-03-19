/** `@type` {import('next').NextConfig} */
const path = require('path');

const nextConfig = {
  // ✅ FIX: Allow massive uploads (2GB limit)
  experimental: {
    serverActions: {
      bodySizeLimit: '2gb',
    },
  },

  // ✅ FIX: Don't bundle native Node.js addons
  serverExternalPackages: ['onnxruntime-node', 'sharp'],

  // Image optimization — cap disk cache to prevent unbounded growth (CVE fix)
  images: {
    maximumDiskCacheSize: 512 * 1024 * 1024, // 512 MB
  },

  // 🚀 PERFORMANCE
  compress: true,
  productionBrowserSourceMaps: false,

  // Fix workspace root warning (monorepo detection)
  outputFileTracingRoot: path.join(__dirname, '../'),

  // API backend proxy - FALLBACK only
  async rewrites() {
    const backendUrl = process.env.BACKEND_URL || 'http://127.0.0.1:8000';
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