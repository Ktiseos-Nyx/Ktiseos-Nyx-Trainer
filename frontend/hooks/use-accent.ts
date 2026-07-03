'use client';

import { useState, useEffect, useCallback } from 'react';
import {
  type AccentColor,
  ACCENT_COLORS,
  DEFAULT_ACCENT,
  STORAGE_KEY,
  getAccentColor,
  setAccentColor,
} from '@/lib/settings';

/**
 * React hook for accent color with cross-tab sync.
 *
 * Reads from localStorage on mount, applies to DOM,
 * and listens for cross-tab changes via the `storage` event.
 */
export function useAccent() {
  const [accent, setAccentState] = useState<AccentColor>(DEFAULT_ACCENT);
  const [mounted, setMounted] = useState(false);

  // Hydrate from localStorage on mount
  useEffect(() => {
    setAccentState(getAccentColor());
    setMounted(true);
  }, []);

  // Apply to DOM whenever accent changes
  useEffect(() => {
    if (!mounted) return;
    setAccentColor(accent);
  }, [accent, mounted]);

  // Cross-tab sync via storage event
  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY) {
        const newColor = e.newValue;
        if (!newColor || newColor === DEFAULT_ACCENT) {
          setAccentState(DEFAULT_ACCENT);
        } else if (ACCENT_COLORS.includes(newColor as AccentColor)) {
          setAccentState(newColor as AccentColor);
        }
      }
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);

  const setAccent = useCallback((color: AccentColor) => {
    setAccentState(color);
  }, []);

  return { accent, setAccent, mounted };
}
