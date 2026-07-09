# Crop Grid — Pagination & Padding Fix

## Current State

`frontend/components/crop/cards/CropGridCard.tsx` renders ALL images in a
scrollable grid but silently truncates at 50 via a hardcoded `maxVisible`.
Users can't access images beyond the first 50. Padding is uneven (manual
`pr-4` compensating for the scrollbar, no bottom padding for the last row).

## Problems

### 1. No pagination (lines 116, 123–124)

```typescript
maxVisible = 50,                              // default prop — hardcoded
const visibleImages = images.slice(0, maxVisible);  // silent truncation
const hasMore = images.length > maxVisible;          // dead-end text
```

- Images beyond 50 are invisible + un-croppable
- The "Showing 50 of X images." text is a dead end — no action to take
- No page controls, no page size selector

### 2. Padding asymmetry (line 149)

```typescript
<div className="grid grid-cols-1 xl:grid-cols-2 gap-4 pr-4">
```

- `pr-4` is a manual offset for the scrollbar — brittle, looks wrong when
  no scrollbar is present (short datasets)
- No `pb-4` — last row of cells presses against the bottom of the
  ScrollArea, visually inconsistent with the top padding from CardContent
- `gap-4` is even between cells but the outer edge padding (CardContent's
  own padding vs `pr-4`) doesn't match

## Fix Plan

### A. Pagination — `CropGridCard` props + state

**New props:**
```typescript
pageSize?: number;        // default 25
```

**Local state (inside component):**
```typescript
const [page, setPage] = useState(0);
const scrollRef = useRef<HTMLDivElement>(null);
```

**Slice logic:**
```typescript
const totalPages = Math.ceil(images.length / pageSize);
const visibleImages = images.slice(page * pageSize, (page + 1) * pageSize);
```

**Controls in CardHeader** (right side, alongside the resolution/aspect badges):
- Page size dropdown: 25 / 50 / 100
- Previous button (disabled on page 0)
- "Page N of M" indicator
- Next button (disabled on last page)
- When page changes, `scrollRef.current.scrollTo(0, 0)`

**Replace the dead-end text:**
```typescript
{images.length > pageSize && (
  <p className="text-center text-sm text-muted-foreground mt-4">
    Showing {Math.min((page + 1) * pageSize, images.length)} of {images.length} images.
  </p>
)}
```
(This updates dynamically with the current page — useful info even without
the pagination controls being visible.)

### B. Padding fixes

```typescript
// BEFORE (line 149):
<div className="grid grid-cols-1 xl:grid-cols-2 gap-4 pr-4">

// AFTER:
<div className="grid grid-cols-1 xl:grid-cols-2 gap-4 pb-4">
```

- Remove `pr-4` — the ScrollArea handles scrollbar offset natively
- Add `pb-4` — consistent bottom gap for the last row
- Top gap already handled by CardContent's padding, so first row is fine

### C. Add `'use client'` to page.tsx if not already

Already has `'use client'` — no change needed.

## Files Changed

| File | Change |
|------|--------|
| `frontend/components/crop/cards/CropGridCard.tsx` | Add pagination (page state, pageSize prop, prev/next controls, page dropdown); fix padding (`pr-4` → `pb-4`); update "Showing X of Y" text |
| `frontend/app/dataset/[name]/crop/page.tsx` | Pass `pageSize` to `CropGridCard` if non-default needed (optional — default 25 is fine) |

## Out of Scope

- **Virtualization** — each cell has an `ImageCropper` with drag/zoom state;
  virtualizing those would be fragile. Pagination (25/page) is sufficient.
- **URL search params** — page state is local-only. Could add `?page=N`
  later if needed, but crop is a single-session workflow.

---

# File Tree — "Open Folder" Should Open a Native OS Dialog

## Current State

`frontend/components/dataset-tools/file-tree.tsx:333` — the "Open Folder"
button (`FolderInput` icon) calls `openPathEditor()` which just toggles
`isEditingPath` to `true`, replacing the header with a text `<Input>` where
users manually type a path. There's no native OS folder picker involved.

```typescript
const openPathEditor = () => {
    setPathInput(settings.currentFolder);
    setIsEditingPath(true);
    setTimeout(() => pathInputRef.current?.select(), 0);
};
```

Users expect "Open Folder" to open the native OS folder-selection dialog.

## Fix Plan

Add a hidden `<input type="file" webkitdirectory>` that triggers the native
folder picker, then extracts the selected path and commits it through the
existing `updateSettings` flow.

### Implementation

1. **Add a hidden file input ref**:
   ```typescript
   const folderInputRef = useRef<HTMLInputElement>(null);
   ```

2. **Add the hidden input** (rendered once, always hidden):
   ```tsx
   <input
     ref={folderInputRef}
     type="file"
     webkitdirectory
     className="hidden"
     onChange={(e) => {
       const file = e.target.files?.[0];
       if (file) {
         // Extract the directory path from the first file's webkitRelativePath
         // e.g. "my_folder/sub/file.png" → path up to "my_folder"
         const path = file.webkitRelativePath.split('/')[0];
         // But we need the full server-side path, not the relative folder name.
         // We send the file list to the backend to resolve.
       }
     }}
   />
   ```

Wait — `webkitdirectory` only gives relative paths, not the full server path.
That's the problem: the backend expects a server-side absolute path like
`/workspace/datasets/my_set`, but the browser only sees the leaf folder name.

**Better approach:** add a dedicated backend endpoint or use existing path
resolution. The flow:

1. User clicks "Open Folder" → hidden `<input webkitdirectory>` opens
2. User selects a folder → we grab one file's `webkitRelativePath`
3. POST the file (or just the folder name) to a new backend endpoint that
   resolves the full path on the server
4. Backend returns the resolved absolute path
5. Frontend calls `updateSettings({ currentFolder: resolvedPath })`

**Alternative — simpler:** keep the text input as a fallback but add a
"Browse..." button that uses the native picker, then resolves via the
existing `/api/dataset-tools/fs` endpoint by searching for the folder name
in the current directory's listing.

**Simplest fix that actually works:** change the "Open Folder" button label
and behavior. Make it clear it's a "Jump to folder" path input (rename the
tooltip to "Go to folder"), and keep it as a quick-navigation tool. This is
honest about what it does without pretending to be a native dialog.

**If we want the real native dialog, the approach is:**

1. Add a hidden `<input ref={folderInputRef} type="file" webkitdirectory className="hidden" />`
2. On "Open Folder" click → `folderInputRef.current?.click()`
3. On file select → extract the folder name from `file.webkitRelativePath`
4. POST to a new endpoint `/api/dataset-tools/resolve-folder` with `{ hint: folderName, baseFolder: settings.currentFolder }`
5. Backend searches `baseFolder` for a directory matching `folderName`, returns the resolved path
6. `updateSettings({ currentFolder: resolvedPath })`

## Files Changed

| File | Change |
|------|--------|
| `frontend/components/dataset-tools/file-tree.tsx` | Add hidden input + native picker logic; or rename button tooltip if going the honest route |
| `api/routes/dataset_tools.py` (or equivalent) | New endpoint `resolve-folder` that finds a folder by name under a base path |

## Caveats

- `webkitdirectory` requires a user gesture (click) — satisfied by the
  button click
- Works in Chrome, Firefox, Edge. No Safari folder-picker support.
- The `resolve-folder` endpoint is a best-effort search; ambiguous names
  return the first match.
