/** @type {import('next').NextConfig} */
const nextConfig = {
  // ✅ FIX: Allow massive uploads (2GB limit)
  // This stops the "Request body exceeded 10MB" error
  experimental: {
    serverActions: {
      bodySizeLimit: '2gb',
    },
  },

  // 🚀 PERFORMANCE: Production optimizations
  output: 'standalone', // Reduces deployment size by 80%+
  compress: true,       // Enable gzip compression
  productionBrowserSourceMaps: false, // Disable source maps in production (saves ~40% size)

  // Fix workspace root warning (monorepo detection)
  outputFileTracingRoot: require('path').join(__dirname, '../'),

  // 📦 BUNDLE ANALYSIS: Uncomment to analyze bundle size
  // webpack: (config, { isServer }) => {
  //   if (!isServer) {
  //     const { BundleAnalyzerPlugin } = require('webpack-bundle-analyzer');
  //     config.plugins.push(new BundleAnalyzerPlugin({ analyzerMode: 'static', openAnalyzer: false }));
  //   }
  //   return config;
  // },

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
