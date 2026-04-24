'use client';

/**
 * Global Error Boundary - catches errors in the root layout itself.
 * This replaces Next.js's generic "Application error" page with
 * something that actually shows what went wrong.
 *
 * NOTE: This must define its own <html>/<body> because the root
 * layout has already failed when this renders.
 */

import { useEffect } from 'react';

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    fetch('/api/debug/client-error', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: error.message,
        digest: error.digest,
        stack: error.stack,
        url: typeof window !== 'undefined' ? window.location.href : 'unknown',
        boundary: 'global',
      }),
    }).catch(() => {});
  }, [error]);

  return (
    <html lang="en">
      <body style={{
        margin: 0,
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontFamily: 'system-ui, -apple-system, sans-serif',
        backgroundColor: '#0a0a0a',
        color: '#e5e5e5',
      }}>
        <div style={{ maxWidth: 600, padding: 32, textAlign: 'center' }}>
          <h1 style={{ fontSize: 24, marginBottom: 8, color: '#f87171' }}>
            Something went wrong
          </h1>
          <p style={{ fontSize: 14, color: '#a3a3a3', marginBottom: 24 }}>
            The application hit an unexpected error. The details below can help with debugging.
          </p>

          {/* Error details - inline styled so it works without Tailwind/globals.css */}
          <pre style={{
            textAlign: 'left',
            backgroundColor: '#1a1a1a',
            border: '1px solid #333',
            borderRadius: 8,
            padding: 16,
            fontSize: 12,
            overflow: 'auto',
            maxHeight: 300,
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}>
            {error.message || 'Unknown error'}
            {process.env.NODE_ENV === 'development' && error.stack && (
              <>
                {'\n\n'}
                {error.stack}
              </>
            )}
            {error.digest && (
              <>
                {'\n\nDigest: '}{error.digest}
              </>
            )}
          </pre>

          <div style={{ marginTop: 24, display: 'flex', gap: 12, justifyContent: 'center' }}>
            <button
              onClick={reset}
              style={{
                padding: '10px 20px',
                borderRadius: 6,
                border: 'none',
                backgroundColor: '#7c3aed',
                color: 'white',
                fontSize: 14,
                cursor: 'pointer',
              }}
            >
              Try again
            </button>
            <button
              onClick={() => window.location.href = '/'}
              style={{
                padding: '10px 20px',
                borderRadius: 6,
                border: '1px solid #333',
                backgroundColor: 'transparent',
                color: '#e5e5e5',
                fontSize: 14,
                cursor: 'pointer',
              }}
            >
              Go home
            </button>
          </div>

          <p style={{ fontSize: 12, color: '#666', marginTop: 24 }}>
            If this keeps happening, please report the issue on GitHub with the error message above.
          </p>
        </div>
      </body>
    </html>
  );
}
