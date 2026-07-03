'use client';

import { Palette } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { useAccent } from '@/hooks/use-accent';
import type { AccentColor } from '@/lib/settings';

const ACCENT_OPTIONS: { color: AccentColor; label: string; dot: string }[] = [
  { color: 'zinc', label: 'Zinc', dot: 'bg-zinc-400' },
  { color: 'red', label: 'Red', dot: 'bg-red-500' },
  { color: 'orange', label: 'Orange', dot: 'bg-orange-500' },
  { color: 'green', label: 'Green', dot: 'bg-green-500' },
  { color: 'blue', label: 'Blue', dot: 'bg-blue-500' },
  { color: 'violet', label: 'Violet', dot: 'bg-violet-500' },
  { color: 'pink', label: 'Pink', dot: 'bg-pink-500' },
];

interface AccentPickerProps {
  className?: string;
}

export function AccentPicker({ className }: AccentPickerProps) {
  const { accent, setAccent, mounted } = useAccent();

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="outline"
          size="icon"
          className={cn('h-9 w-9', className)}
          aria-label="Change accent color"
        >
          <Palette className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-48">
        <div className="px-2 py-1.5">
          <p className="text-xs font-medium text-muted-foreground mb-2">Accent Color</p>
          <div className="flex gap-1.5">
            {ACCENT_OPTIONS.map((opt) => (
              <button
                key={opt.color}
                onClick={() => setAccent(opt.color)}
                className={cn(
                  'h-7 w-7 rounded-full border-2 transition-all hover:scale-110',
                  opt.dot,
                  mounted && accent === opt.color
                    ? 'border-foreground scale-110 ring-2 ring-foreground/20'
                    : 'border-transparent',
                )}
                aria-label={opt.label}
                title={opt.label}
              />
            ))}
          </div>
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
