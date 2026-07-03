'use client';

import { useControllableState } from '@radix-ui/react-use-controllable-state';
import { Monitor, Moon, Palette, Sun } from 'lucide-react';
import { motion } from 'motion/react';
import { useCallback, useEffect, useState } from 'react';
import { cn } from '@/lib/utils';
import { useAccent } from '@/hooks/use-accent';
import type { AccentColor } from '@/lib/settings';

const themes = [
  { key: 'system', icon: Monitor, label: 'System theme' },
  { key: 'light', icon: Sun, label: 'Light theme' },
  { key: 'dark', icon: Moon, label: 'Dark theme' },
];

const ACCENT_OPTIONS: { color: AccentColor; label: string; dot: string }[] = [
  { color: 'zinc', label: 'Zinc', dot: 'bg-zinc-400' },
  { color: 'red', label: 'Red', dot: 'bg-red-500' },
  { color: 'orange', label: 'Orange', dot: 'bg-orange-500' },
  { color: 'green', label: 'Green', dot: 'bg-green-500' },
  { color: 'blue', label: 'Blue', dot: 'bg-blue-500' },
  { color: 'violet', label: 'Violet', dot: 'bg-violet-500' },
  { color: 'pink', label: 'Pink', dot: 'bg-pink-500' },
];

export type ThemeSwitcherProps = {
  value?: 'light' | 'dark' | 'system';
  onChange?: (theme: 'light' | 'dark' | 'system') => void;
  defaultValue?: 'light' | 'dark' | 'system';
  className?: string;
};

export const ThemeSwitcher = ({
  value,
  onChange,
  defaultValue = 'system',
  className,
}: ThemeSwitcherProps) => {
  const [theme, setTheme] = useControllableState({
    defaultProp: defaultValue,
    prop: value,
    onChange,
  });
  const [mounted, setMounted] = useState(false);
  const { accent, setAccent, mounted: accentMounted } = useAccent();

  const handleThemeClick = useCallback(
    (themeKey: 'light' | 'dark' | 'system') => {
      setTheme(themeKey);
    },
    [setTheme]
  );

  useEffect(() => {
    const timer = setTimeout(() => setMounted(true), 0);
    return () => clearTimeout(timer);
  }, []);

  if (!mounted) {
    return null;
  }

  return (
    <div className={cn('flex items-center gap-2', className)}>
      <div
        className={cn(
          'relative isolate flex h-8 rounded-full bg-background p-1 ring-1 ring-border'
        )}
      >
        {themes.map(({ key, icon: Icon, label }) => {
          const isActive = theme === key;

          return (
            <button
              aria-label={label}
              className="relative h-6 w-6 rounded-full"
              key={key}
              onClick={() => handleThemeClick(key as 'light' | 'dark' | 'system')}
              type="button"
            >
              {isActive && (
                <motion.div
                  className="absolute inset-0 rounded-full bg-secondary"
                  layoutId="activeTheme"
                  transition={{ type: 'spring', duration: 0.5 }}
                />
              )}
              <Icon
                className={cn(
                  'relative z-10 m-auto h-4 w-4',
                  isActive ? 'text-foreground' : 'text-muted-foreground'
                )}
              />
            </button>
          );
        })}
      </div>
      <div className="flex h-8 items-center gap-1 rounded-full bg-background px-2 ring-1 ring-border">
        <Palette className="h-3.5 w-3.5 text-muted-foreground" />
        {ACCENT_OPTIONS.map((opt) => (
          <button
            key={opt.color}
            onClick={() => setAccent(opt.color)}
            className={cn(
              'h-5 w-5 rounded-full border-2 transition-all hover:scale-110',
              opt.dot,
              accentMounted && accent === opt.color
                ? 'border-foreground scale-110 ring-1 ring-foreground/20'
                : 'border-transparent',
            )}
            aria-label={opt.label}
            title={opt.label}
            type="button"
          />
        ))}
      </div>
    </div>
  );
};
