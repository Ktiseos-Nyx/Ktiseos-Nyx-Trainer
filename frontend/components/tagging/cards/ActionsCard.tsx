/**
 * Actions Card — Start/Stop controls + live progress.
 *
 * Presentational only; the page owns the job state and passes it down. Lives
 * outside the tabs so it stays visible no matter which settings tab is open.
 */

'use client';

import { Play, Square } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';

interface ActionsCardProps {
  tagging: boolean;
  canStart: boolean;
  progress: number;
  currentImage: string | null;
  totalImages: number | null;
  onStart: () => void;
  onStop: () => void;
}

export function ActionsCard({
  tagging,
  canStart,
  progress,
  currentImage,
  totalImages,
  onStart,
  onStop,
}: ActionsCardProps) {
  return (
    <Card>
      <CardContent className="pt-6 space-y-4">
        <div className="flex gap-3">
          <Button onClick={onStart} disabled={tagging || !canStart} className="flex-1 gap-2">
            <Play className="h-5 w-5" />
            {tagging ? 'Tagging…' : 'Start'}
          </Button>
          {tagging && (
            <Button onClick={onStop} variant="destructive" className="gap-2">
              <Square className="h-5 w-5" />
              Stop
            </Button>
          )}
        </div>

        {tagging && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="font-medium">Progress: {progress}%</span>
              {totalImages != null && <span className="text-muted-foreground">{totalImages} images</span>}
            </div>
            <Progress value={progress} />
            {currentImage && <p className="text-xs text-muted-foreground truncate">{currentImage}</p>}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
