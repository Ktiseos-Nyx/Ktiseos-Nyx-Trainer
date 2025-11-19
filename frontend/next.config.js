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
};

module.exports = nextConfig;
