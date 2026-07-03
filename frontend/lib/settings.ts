/**
 * Accent color type and localStorage persistence.
 *
 * Accent colors are stored in localStorage under key "app-accent".
 * This is separate from next-themes (which handles light/dark).
 */

export type AccentColor = 'zinc' | 'red' | 'orange' | 'green' | 'blue' | 'violet' | 'pink';

export const ACCENT_COLORS: AccentColor[] = ['zinc', 'red', 'orange', 'green', 'blue', 'violet', 'pink'];

export const DEFAULT_ACCENT: AccentColor = 'zinc';

export const STORAGE_KEY = 'app-accent';

/** Get the stored accent color from localStorage. */
export function getAccentColor(): AccentColor {
  if (typeof window === 'undefined') return DEFAULT_ACCENT;
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored && ACCENT_COLORS.includes(stored as AccentColor)) {
      return stored as AccentColor;
    }
  } catch {
    // localStorage unavailable
  }
  return DEFAULT_ACCENT;
}

/** Save accent color to localStorage and apply to DOM. */
export function setAccentColor(color: AccentColor): void {
  if (typeof window === 'undefined') return;
  try {
    if (color === DEFAULT_ACCENT) {
      localStorage.removeItem(STORAGE_KEY);
      document.documentElement.removeAttribute('data-accent');
    } else {
      localStorage.setItem(STORAGE_KEY, color);
      document.documentElement.setAttribute('data-accent', color);
    }
  } catch {
    // localStorage unavailable
  }
}
