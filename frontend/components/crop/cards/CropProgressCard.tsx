'use client';

import { Loader2, StopCircle, CheckCircle2, AlertCircle } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import type { CropJobStatus } from '@/lib/api';

interface CropProgressCardProps {
  isRunning: boolean;
  status: CropJobStatus | null;
  logs: string[];
  onStart: () => void;
  onStop: () => void;
  canStart: boolean;
}

export function CropProgressCard({
  isRunning,
  status,
  logs,
  onStart,
  onStop,
  canStart,
}: CropProgressCardProps) {
  const progress = status?.progress ?? 0;
  const isComplete = status?.status === 'completed';
  const isFailed = status?.status === 'failed';
  const isCancelled = status?.status === 'cancelled';
  const isIdle = !isRunning && !isComplete && !isFailed && !isCancelled;

  return (
    <Card className="border-amber-500/30">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          {isRunning && <Loader2 className="h-5 w-5 text-amber-400 animate-spin" />}
          {isComplete && <CheckCircle2 className="h-5 w-5 text-green-400" />}
          {(isFailed || isCancelled) && <AlertCircle className="h-5 w-5 text-red-400" />}
          {isIdle && <Loader2 className="h-5 w-5 text-muted-foreground" />}
          Crop Progress
        </CardTitle>
        <CardDescription>
          {isIdle && 'Ready to start cropping'}
          {isRunning && `Cropping... ${status?.cropped_files ?? 0}/${status?.total_files ?? 0}`}
          {isComplete && 'Crop completed'}
          {isFailed && `Failed: ${status?.errors?.[0] ?? 'Unknown error'}`}
          {isCancelled && 'Crop cancelled'}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Progress Bar */}
        {(isRunning || isComplete || isFailed || isCancelled) && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Progress</span>
              <span className="font-mono">{progress}%</span>
            </div>
            <Progress value={progress} className="h-2" />
            {status?.current_file && (
              <p className="text-xs text-muted-foreground truncate">
                Processing: {status.current_file}
              </p>
            )}
          </div>
        )}

        {/* Log Output */}
        {logs.length > 0 && (
          <div className="rounded-md bg-muted p-3 max-h-[200px] overflow-y-auto font-mono text-xs space-y-0.5">
            {logs.slice(-20).map((log, i) => (
              <div
                key={i}
                className={log.startsWith('ERROR') ? 'text-red-400' : 'text-muted-foreground'}
              >
                {log}
              </div>
            ))}
          </div>
        )}

        {/* Actions */}
        <div className="flex gap-2">
          {isIdle && (
            <Button onClick={onStart} disabled={!canStart} className="gap-1.5">
              <Loader2 className="h-4 w-4" />
              Start Crop
            </Button>
          )}
          {isRunning && (
            <Button onClick={onStop} variant="destructive" className="gap-1.5">
              <StopCircle className="h-4 w-4" />
              Stop
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
