import { NextRequest, NextResponse } from 'next/server';

/**
 * Middleware that logs all API requests handled by Next.js.
 * This provides visibility into what the Node.js side is doing,
 * especially on local Windows dev where FastAPI sees very little traffic.
 *
 * Logs go to the terminal (stdout) where `npm run dev` or `npm start` is running.
 */
export function middleware(request: NextRequest) {
  const start = Date.now();
  const { method, nextUrl } = request;
  const path = nextUrl.pathname;

  // Log the incoming request
  const timestamp = new Date().toISOString().slice(11, 23);
  console.log(`[${timestamp}] ${method} ${path}`);

  const response = NextResponse.next();

  // Log duration after response is ready
  const duration = Date.now() - start;
  if (duration > 100) {
    console.log(`[${timestamp}] ${method} ${path} (${duration}ms - slow)`);
  }

  return response;
}

// Only run middleware on API routes — don't slow down page navigation
export const config = {
  matcher: '/api/:path*',
};
