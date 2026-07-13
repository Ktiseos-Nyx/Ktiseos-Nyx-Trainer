# Arc En Ciel Downloader — Pagination, Manual Data, Modal Overflow

## Current State

`frontend/app/models/arcenciel/page.tsx` — single-page model browser with
infinite scroll, a versions dialog, and no way to manually add models the
API doesn't know about.

---

## 1. Infinite Scroll → Pagination

### Current (lines 103–116)

Uses `IntersectionObserver` on the last card to auto-fetch more:

```typescript
const lastModelRef = useCallback((node: HTMLDivElement | null) => {
    if (loading) return;
    if (observer.current) observer.current.disconnect();
    observer.current = new IntersectionObserver((entries) => {
        if (entries[0].isIntersecting && hasMore) {
            setPage((prev) => prev + 1);
        }
    });
    if (node) observer.current.observe(node);
}, [loading, hasMore]);
```

Problems:
- No page controls — can't skip ahead or go back
- Auto-loads on scroll, which is unpredictable with large result sets
- Loss of scroll position if navigating away and back

### Plan

Replace the observer with explicit pagination controls:

1. **Remove** `observer`, `lastModelRef`, the `useRef<IntersectionObserver>`
2. **Keep** `page`, `hasMore`, `loadModels` — they already support page numbers
3. **Add pagination bar** below the model grid:
   ```
   [← Previous]  Page 3 of 12  [Next →]
   ```
4. **Page size dropdown** (20 / 40 / 80) stored in local state, passed as
   `limit` to `sourcesAPI.search()`
5. **Update the "Showing X models" text** to reflect current page:
   ```
   Showing 41–60 of 237 models
   ```

### Files Changed

| File | Change |
|------|--------|
| `frontend/app/models/arcenciel/page.tsx` | Remove IntersectionObserver, add pagination bar + page size selector |

---

## 2. Manual Metadata Editing (fields the API doesn't return)

### Problem

The Arc En Ciel API returns basic fields (`type` = CHECKPOINT/LORA/VAE/OTHER,
`base_model` = SD1.5/SDXL/etc.), but **doesn't** return many useful metadata
fields that models may have (e.g. on Civitai):

| Missing Field | Where it'd live |
|---------------|-----------------|
| **LoRA subtype** (LoCon / LoHa / LoKR / LoRA) | `SourceModelVersion` or `SourceModelDetail` |
| **Trigger words** | `SourceModelVersion` |
| **Trained tags / concepts** | `SourceModelDetail` |
| **Training info** (resolution, repeats, etc.) | `SourceModelVersion` |
| **Usage tips** / recommended settings | `SourceModelDetail` |
| **Download count** / popularity stats | `SourceModelSummary` |
| **Rating / review score** | `SourceModelSummary` |
| **Multiple files per version** (some versions ship multiple .safetensors) | `SourceModelVersion` |

These fields exist in the Civitai API response but Arc En Ciel's API proxy
doesn't forward them.

### Plan

Add an **inline metadata editor** to the versions dialog so users can fill
in the gaps per model:

1. **Edit button** on each version card (pencil icon) — opens the version's
   metadata for editing
2. **Inline edit panel** (expandable, or a small sub-dialog) with fields:
   - LoRA subtype dropdown: LoRA / LoCon / LoHa / LoKR (only shown when
     `type === 'LORA'`)
   - Trigger words (tag-style input, comma-separated)
   - Trained tags (tag input)
   - Notes / usage tips (textarea)
3. **Local storage** — save edits to `localStorage` keyed by
   `source-model_id-version_id` so they persist across sessions. The backend
   doesn't need to store these; they're annotations local to this browser.
4. **Visual indicator** — show a small dot/badge on version cards that have
   manual annotations so you can tell what's been filled in

### Future Extension

If the backend eventually adds these fields to the API response, the local
storage values should be treated as overrides (local wins) rather than
replacements.

### Files Changed

| File | Change |
|------|--------|
| `frontend/app/models/arcenciel/page.tsx` | Add metadata editor UI to version cards; save/load from localStorage |
| `frontend/lib/api.ts` | No API changes needed — purely frontend-side annotation |

---

## 3. Versions Dialog — Download Button Scrolling

### Problem (line 517)

The dialog is a single scrollable container (`max-h-[85vh] overflow-y-auto`):
```tsx
<DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto">
```

For models with many versions, users must scroll through the entire version
list. The download button is on each version card, so it's not hidden — but
on models with 20+ versions, the cards push each other down and you scroll
a lot to find a specific version.

Additionally, if a model has a long HTML description (line 529–533), the
description area expands significantly, pushing the version list further
down.

### Plan

Two-step fix:

**A. Collapsible description** — truncate the description to 3 lines with a
"Show more" toggle instead of `line-clamp-5`. The description occupies
vertical real estate before users even reach the version list.

**B. Version list max-height with internal scroll** — constrain the versions
section within the dialog so its height is capped, and it scrolls
independently:

```tsx
{/* Versions */}
<div className="space-y-3">
  <h4 className="font-semibold text-foreground shrink-0">Versions</h4>
  {selectedModel.versions.length === 0 ? (
    <p className="text-sm text-muted-foreground">No versions available</p>
  ) : (
    <div className="max-h-[40vh] overflow-y-auto space-y-3 pr-1">
      {selectedModel.versions.map((version) => ( ... ))}
    </div>
  )}
</div>
```

This keeps the dialog header, model info badges, and footer always visible
while the version list scrolls independently. The download buttons are
always accessible within each compact version card.

**C. Sticky model info bar** — keep the model type/base/NSFW badges visible
as a sticky row between the title and the version list (already in a
`flex-wrap` div at line 542, but it scrolls away). Making it `sticky top-0`
inside the dialog keeps context visible.

### Files Changed

| File | Change |
|------|--------|
| `frontend/app/models/arcenciel/page.tsx` | Collapsible description; constrain version list to `max-h-[40vh]` with internal scroll; sticky model info badges |
