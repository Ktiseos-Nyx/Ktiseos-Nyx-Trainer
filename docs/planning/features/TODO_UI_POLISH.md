# UI Polish & Pending Fixes

## 2025-11-30 - Settings Page "Advanced" Toggle - RESOLVED ✅

**Decision:** Enable all advanced parameters by default for alpha testing

**What was done:**
- Changed default values in `useSettings.ts` for all visibility flags to `true`:
  - `showAdvancedLycoris: true`
  - `showBlockwiseLR: true`
  - `showSD2Params: true`
  - `showPerformanceTuning: true`
  - `showExperimentalFeatures: true`

**Result:** All advanced training parameters are now visible by default in Training Config

**Future:** Can revisit adding toggles to Settings later if needed, but for alpha we want maximum visibility

---

## Tag Editor - Images Not Displaying - RESOLVED ✅

**Issue:** Tag editor page showed correct image count but images didn't display in the grid

**Root Cause:** Frontend had placeholder div instead of actual `<img>` tag

**Fix Applied (2025-11-30):**
- Replaced placeholder with actual `<img>` tag using `/api/files/image/{path}` endpoint
- Added proper image URL construction: `/api/files/image${img.image_path.substring(1)}`
- Added error handling with fallback to placeholder on image load failure
- Images now display correctly in the tag editor grid

**Status:** ✅ FIXED - Images now display properly in frontend/app/dataset/tags/page.tsx:321-335

---

## Docs Page Layout

**Fixed:**
- List items now properly render HTML with `dangerouslySetInnerHTML`
- Added `break-words` and `overflow-hidden` to prevent content overflow
- Added `break-all` to inline code blocks
- Fixed sections sidebar with `overflow-hidden` and text truncation

**Status:** ✅ Fixed - page should no longer be skewed
