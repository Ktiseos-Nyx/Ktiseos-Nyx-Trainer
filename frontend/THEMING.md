# Theming Documentation

## Current Status: DARK MODE ONLY (By Design)

**Date:** 2025-11-20
**Status:** Theme toggle exists in navbar but pages are intentionally dark-styled

## The Problem

The app has a theme toggle in the navbar that works, but:
- Pages use hardcoded dark gradients: `bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900`
- Text uses hardcoded gray colors: `text-gray-300`, `text-gray-400`
- Components use dark backgrounds: `bg-slate-800/50`, `border-slate-700`

When you toggle to light mode, **only the navbar changes** - pages stay dark.

## What We Tried (DON'T DO THIS AGAIN!)

### ❌ Attempt 1: Use Theme Variables (FAILED)

**Commit:** 4b0e471 (Later reverted in e305ed5)

Changed:
- `bg-slate-950` → `bg-background`
- `text-gray-300` → `text-muted-foreground`
- `border-slate-700` → `border-border`

**Why it failed:**
- Forced **black text** on animated gradient backgrounds (unreadable)
- Forced **yellow text** on yellow alert banners (invisible)
- Theme variables don't work well with custom gradient backgrounds
- Lost all the intentional styling

**Lesson:** Theme variables are meant for simple backgrounds/text, not styled components with gradients and specific color schemes.

## The Right Approach (If We Want Light Mode)

### Option 1: Use `dark:` Prefix Classes

Instead of replacing colors, **add** light mode variants:

```tsx
// WRONG (what we tried):
className="bg-slate-900"
// Becomes:
className="bg-background"  // ❌ Breaks styling

// RIGHT:
className="bg-slate-900"
// Becomes:
className="bg-white dark:bg-slate-900"  // ✅ Works for both
```

For the gradient backgrounds:
```tsx
// Current:
className="bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900"

// Light mode support:
className="bg-gradient-to-br from-slate-50 via-purple-100 to-slate-50 dark:from-slate-900 dark:via-purple-900 dark:to-slate-900"
```

For text:
```tsx
// Current:
className="text-gray-300"

// Light mode support:
className="text-gray-900 dark:text-gray-300"
```

### Option 2: Keep Dark Mode Only

Most developer/training tools are dark mode focused. If light mode isn't a priority:
1. Remove the theme toggle from navbar
2. Set `defaultTheme="dark"` and `enableSystem={false}` in layout.tsx
3. Keep all current styling as-is

## Accessibility Note

Light/dark mode IS important for accessibility:
- Photosensitivity
- Migraines
- Visual processing disorders
- Eye strain
- Different lighting conditions

**Not** just a developer preference!

## Current Theme Setup

- **ThemeProvider:** `/frontend/app/layout.tsx` (line 20-25)
- **Theme Toggle:** `/frontend/components/blocks/navigation/navbar.tsx` (uses ThemeSwitcher component)
- **CSS Variables:** `/frontend/app/globals.css` (lines 6 and 47 define `--background`)

## Pages That Need Light Mode Support

If we decide to properly support light mode, these need `dark:` classes:

1. `/frontend/app/calculator/page.tsx` - Line 66 (gradient background)
2. `/frontend/app/models/page.tsx` - Line 98 (gradient background)
3. `/frontend/app/utilities/page.tsx` - Line 12 (gradient background)
4. `/frontend/components/blocks/navigation/footer.tsx` - Lines 10, 14, 30, 42, 51, 60 (dark backgrounds and text)

Plus many other pages we haven't audited yet.

## Recommendation

**For now:** Document that the app is dark mode only and remove/disable the theme toggle to avoid user confusion.

**For later:** If light mode is needed for accessibility, do a proper audit and add `dark:` classes systematically across ALL pages and components (this is a multi-hour task for a 700+ line codebase).

---

**Last Updated:** 2025-11-20
**Last Modified By:** Claude (Session with Belmont)
