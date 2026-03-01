/**
 * Debug endpoint: /api/debug/memory
 * Returns current Node.js process memory usage and uptime.
 * Useful for monitoring heap growth on long-running VastAI instances.
 */

import { NextResponse } from 'next/server';

export async function GET() {
  const mem = process.memoryUsage();
  const uptimeSeconds = process.uptime();

  return NextResponse.json({
    heapUsed: `${Math.round(mem.heapUsed / 1024 / 1024)} MB`,
    heapTotal: `${Math.round(mem.heapTotal / 1024 / 1024)} MB`,
    rss: `${Math.round(mem.rss / 1024 / 1024)} MB`,
    external: `${Math.round(mem.external / 1024 / 1024)} MB`,
    uptime: `${Math.floor(uptimeSeconds / 3600)}h ${Math.floor((uptimeSeconds % 3600) / 60)}m`,
    uptimeSeconds: Math.round(uptimeSeconds),
  });
}
