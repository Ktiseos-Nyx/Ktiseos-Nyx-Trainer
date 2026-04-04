'use client';

/**
 * Route-level Error Boundary - catches errors in page components
 * while keeping the root layout (navbar, footer, theme) intact.
 */

import { Button } from '@/components/ui/button';

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="container mx-auto flex flex-col items-center justify-center min-h-[60vh] p-8 text-center">
      <h2 className="text-2xl font-bold text-destructive mb-2">
        Something went wrong
      </h2>
      <p className="text-muted-foreground mb-6 max-w-md">
        This page hit an error. The navbar and other pages should still work.
      </p>

      {/* Show error details for debugging */}
      <pre className="text-left bg-muted/50 border rounded-lg p-4 text-xs overflow-auto max-h-64 max-w-2xl w-full mb-6 whitespace-pre-wrap break-words">
        {error.message || 'Unknown error'}
        {error.digest && `\n\nDigest: ${error.digest}`}
      </pre>

      <div className="flex gap-3">
        <Button onClick={reset} variant="default">
          Try again
        </Button>
        <Button onClick={() => window.location.href = '/'} variant="outline">
          Go home
        </Button>
      </div>
    </div>
  );
}
