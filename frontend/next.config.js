/** @type {import('next').NextConfig} */
const nextConfig = {
  // API backend proxy
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: process.env.BACKEND_URL || 'http://localhost:8000/api/:path*',
      },
    ];
  },
  // Disable strict mode for development (optional)
  reactStrictMode: true,
  // Output standalone for Docker
  output: 'standalone',
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
