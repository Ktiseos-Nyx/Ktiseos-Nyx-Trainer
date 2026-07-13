# Crop — Inline Redesign (No Dialog)

**Status:** Draft  
**Priority:** High  
**Last updated:** 2026-07-10

## Goal

Replace the current crop overlay (dialog + diceUI Cropper) with an **inline crop
experience** — no dialog, no modal. Each card in the grid shows a **thumbnail +
inline Cropper** directly, à la Birme.net.

## What We Have Now

| Component | What it does | File |
|-----------|-------------|------|
| `CropGridCard` | Compact thumbnail grid (2-5 cols), pagination, click → overlay | `components/crop/cards/CropGridCard.tsx` |
| diceUI `Cropper` | Fixed-window birme-style cropper (move image behind crop frame) | `components/ui/cropper.tsx` (1788 lines) |
| `thumbToOriginal` | Scales 768px thumbnail coords → original image coords (inverts sharp's fit:cover) | `app/dataset/[name]/crop/page.tsx` |
| `handleSaveCrop` | Canvas-generated cropped preview (192px data URL) | `CropGridCard.tsx` |
| `imageDims` Map | Original image width × height per image | `page.tsx` (fetched via `/api/dataset-tools/image-dims`) |

## Problem With Current Dialog Approach

1. **Dialog UX friction** — click to open, click to close, can't compare images side-by-side
2. **square thumbnail pre-crop** — sharp's `fit:'cover' centre-crop` means the thumbnail is already cropped, requiring complex `thumbToOriginal` math to undo
3. **Save state lost on re-open** — `onCropChange` nullifies `latestPixelCropRef` before `onCropComplete` fires, requiring fallback logic
4. **Coordinate space mismatch** — Cropper operates in 768px thumbnail space but backend expects original pixel coords

## The Target (Birme-Style)

```
┌──────────────────────────────────────────┐
│  [Card 1]        [Card 2]        [Card 3] │
│ ┌──────────┐   ┌──────────┐   ┌────────┐ │
│ │          │   │  ╔══════╗│   │        │ │
│ │  img_01  │   │  ║ crop ║│   │ img_03 │ │
│ │          │   │  ╚══════╝│   │        │ │
│ │  [✂️]    │   │  [💾][↩️] │   │ [✂️]    │ │
│ └──────────┘   └──────────┘   └────────┘ │
└──────────────────────────────────────────┘
```

- Each card: thumbnail at **original aspect ratio** (NOT square)
- Click "✂️ Crop" → card expands, Cropper overlays the image
- Only **one card in crop mode** at a time (performance)
- Save/Reset buttons live on the card itself
- Cropped preview replaces the thumbnail immediately after save

## Available Components to Use

| Component | Why |
|-----------|-----|
| `BentoGrid` (`components/ui/bento-grid.tsx`) | `auto-rows` grid, cards can span multiple rows when expanded |
| `Cluster` (`components/ui/cluster.tsx`) | Flex-based grouping, could wrap cards |
| `Cropper` + `CropperImage` + `CropperArea` | Already in codebase, proven to work |
| `reac-image-crop` npm (^11.1.2) | Alternative if diceUI Cropper is too heavy per-card |
| `vaul` (^1.1.2) | Drawer component if we ever want a bottom-sheet variant |

## Key Design Decisions Needed

### 1. Thumbnail aspect ratio
**Current**: 192×192 square (sharp `fit:'cover'`) — pre-cropped
**Target**: auto-ratio (192 × auto) using `fit:'inside'` — preserves original shape, no pre-crop
**Impact**: eliminates the complex `thumbToOriginal` math — coordinates map 1:1 scaled

### 2. Per-card vs 1-at-a-time
**Per-card Cropper** — too slow (canvas rendering × 25+)
**1-at-a-time** ✅ — only the active card mounts a Cropper; others show static thumbnails

### 3. Expand-in-place vs inline overlay
**Expand-in-place** — the card grows to span more grid rows, Cropper renders inside the expanded area
**Inline overlay** — Cropper overlays the card without layout shift (absolute positioning within the card)

### 4. Grid container
**BentoGrid** — `auto-rows minmax(180px,1fr)` with 3 cols, cards can span rows
**Plain CSS grid** — more control, simpler CSS, less abstraction

## Build Order

### Phase A — Thumbnail ratio fix (prerequisite)
- [ ] A.1 Change `size=192` thumbnail to use `fit:'inside'` instead of `fit:'cover'` (or generate at varied sizes)
- [ ] A.2 OR use raw `/api/dataset/serve/` endpoint and handle display sizing in CSS
- [ ] A.3 Grid cards display at original aspect ratio with `object-fit: contain`

### Phase B — Inline Cropper (core feature)
- [ ] B.1 Add `activeCropImage` state (which card is in crop mode)
- [ ] B.2 Only mount Cropper on the active card — others show static thumbnails
- [ ] B.3 Cropper overlays the thumbnail inline via absolute positioning or card expansion
- [ ] B.4 Save/Cancel buttons on the card (not in dialog header/footer)
- [ ] B.5 On Save: generate cropped preview, swap thumbnail, close crop mode
- [ ] B.6 On Cancel: revert to original thumbnail, close crop mode

### Phase C — Polish
- [ ] C.1 Visual indicator on cards with saved crops (already have violet dot)
- [ ] C.2 Cropped preview shows immediately on save (already have canvas flow)
- [ ] C.3 Coordinate math simplified — no more `thumbToOriginal` if thumbnails aren't square

## Out of Scope
- Per-image zoom/pan memory across mode switches (nice-to-have, not blocking)
- Batch "apply same crop to all images" (Phase 4 feature)
- Keyboard-only crop mode (accessibility, Phase 4)

## Reference
- `documentation/planning/crop-grid-pagination-padding.md` — original crop page plan
- `frontend/components/ui/cropper.tsx` — diceUI Cropper (~1788 lines)
- `frontend/components/ui/bento-grid.tsx` — BentoGrid candidate
- `frontend/components/masonry-grid.tsx` — MasonryGrid (text-only, not for images)
- `frontend/components/ui/cluster.tsx` — Cluster layout primitive
