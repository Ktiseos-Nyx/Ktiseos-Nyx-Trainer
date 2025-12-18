/** @type {import('next').NextConfig} */
const nextConfig = {
  // ✅ FIX: Allow massive uploads (2GB limit)
  // This stops the "Request body exceeded 10MB" error
  experimental: {
    serverActions: {
      bodySizeLimit: '2gb',
    },
  },

  // API backend proxy
  async rewrites() {
    // Keep this as 127.0.0.1!
    // Since Next.js and Python run in the same container on Vast,
    // they talk via localhost internally.
    const backendUrl = process.env.BACKEND_URL || 'http://127.0.0.1:8000';

    return [
      {
        source: '/api/:path*',
        destination: `${backendUrl}/api/:path*`,
      },
    ];
  },

  reactStrictMode: true,
  eslint: { ignoreDuringBuilds: true },
  typescript: { ignoreBuildErrors: true },
};

module.exports = nextConfig;
