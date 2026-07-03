/**
 * Convert Progress Card — live progress bar + stop control.
 */

'use client';

import { Play, Square, CheckCircle, XCircle } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';

interface ConvertProgressCardProps {
  isRunning: boolean;
  progress: number;
  totalFiles: number;
  convertedFiles: number;
  currentFile: string | null;
  isComplete: boolean;
  hasError: boolean;
  result?: {
    success: boolean;
    converted: number;
    total: number;
    errors: string[];
    output_dir: string;
    warning?: string;
  } | null;
  onStart: () => void;
  onStop: () => void;
}

export function ConvertProgressCard({
  isRunning,
  progress,
  totalFiles,
  convertedFiles,
  currentFile,
  isComplete,
  hasError,
  result,
  onStart,
  onStop,
}: ConvertProgressCardProps) {
  return (
    <Card>
      <CardContent className="pt-6 space-y-4">
        {/* Action Buttons */}
        <div className="flex gap-3">
          <Button
            onClick={onStart}
            disabled={isRunning}
            className="flex-1 gap-2"
          >
            <Play className="h-5 w-5" />
            {isRunning ? 'Converting…' : 'Start Conversion'}
          </Button>
          {isRunning && (
            <Button onClick={onStop} variant="destructive" className="gap-2">
              <Square className="h-5 w-5" />
              Stop
            </Button>
          )}
        </div>

        {/* Progress Bar */}
        {isRunning && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="font-medium">Progress: {progress}%</span>
              <span className="text-muted-foreground">
                {convertedFiles}/{totalFiles} files
              </span>
            </div>
            <Progress value={progress} />
            {currentFile && (
              <p className="text-xs text-muted-foreground truncate">
                Converting: {currentFile}
              </p>
            )}
          </div>
        )}

        {/* Completion Status */}
        {isComplete && result && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm">
              {result.success ? (
                <CheckCircle className="h-4 w-4 text-green-500" />
              ) : (
                <XCircle className="h-4 w-4 text-red-500" />
              )}
              <span className={result.success ? 'text-green-600' : 'text-red-600'}>
                {result.success
                  ? `Converted ${result.converted}/${result.total} images`
                  : 'Conversion failed'}
              </span>
            </div>
            {result.output_dir && (
              <p className="text-xs text-muted-foreground">
                Output: <span className="font-mono">{result.output_dir}</span>
              </p>
            )}
            {result.warning && (
              <p className="text-xs text-amber-600">{result.warning}</p>
            )}
            {result.errors.length > 0 && (
              <div className="text-xs text-red-600 space-y-1">
                <p>{result.errors.length} file(s) failed:</p>
                <ul className="list-disc list-inside max-h-20 overflow-y-auto">
                  {result.errors.slice(0, 5).map((err, i) => (
                    <li key={i} className="truncate">{err}</li>
                  ))}
                  {result.errors.length > 5 && (
                    <li>…and {result.errors.length - 5} more</li>
                  )}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* Error State */}
        {hasError && !isComplete && (
          <div className="flex items-center gap-2 text-sm text-red-600">
            <XCircle className="h-4 w-4" />
            <span>Conversion failed. Check logs for details.</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
