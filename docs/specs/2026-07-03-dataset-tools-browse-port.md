# Dataset-Tools "Browse" View — Port into the Trainer (Slice 1)

**Status:** Draft for review
**Date:** 2026-07-03
**Parent:** BETA_PLANNING §11.2 (DT integration) + §21. First of several DT slices.

---

## 1. Goal

Bring Dataset-Tools' **Browse view** into Ktiseos-Nyx-Trainer as an in-app page at
`/dataset-tools`, reached from the trainer's existing **Dataset menu**. Users get
DT's file browser + image preview + metadata / safetensors inspector *inside* the
trainer — no second app, no separate process (both are Next 16 / React 19 / shadcn).

**Guiding rule (Dusk):** *replicate DT's current layout verbatim, with exactly two
changes* — (1) no DT navbar (the trainer's navbar hosts it), (2) delete the redundant
bottom thumbnail drawer. This is a faithful port, not a redesign. **Cherry-pick the
files this view needs; leave DT's unused UI-kit graveyard behind.**

## 2. Layout

Source of truth: DT `app/page.tsx`. Reproduce it, with the two deltas marked ✂️.

```
Trainer navbar  →  Dataset ▾  →  "Dataset Tools"        ✂️ DT's own navbar NOT ported

/dataset-tools page:
┌ toolbar ──────────────────────────────────────────────────────────┐
│  [ List | Thumbnail ] viewMode toggle          [ ⇤ metadata ] │
├──────────────┬──────────────────────────────┬─────────────────────┤
│ FileTree     │   ImagePreview               │  Metadata  (image)  │
│ (list OR     │   (center — full height)     │   OR                │
│  thumbnail   │                              │  Safetensors        │
│  per toggle) │   ✂️ bottom ThumbnailViewport │  (.safetensors)     │
│  15%         │      drawer REMOVED          │  25%, toggleable    │
└──────────────┴──────────────────────────────┴─────────────────────┘
```

**Deltas vs DT `page.tsx`:**
1. **Remove** the `id="thumbnails"` `Panel` + its `ThumbnailViewport` (`page.tsx:266-274`)
   and the surrounding vertical `PanelGroup` — the center becomes a single
   `ImagePreview` panel. Thumbnail browsing survives via the FileTree's `viewMode`
   toggle (unchanged).
2. **No DT navbar** — the page renders inside the trainer's existing chrome.

Everything else (toolbar toggle, FileTree `viewMode`, show/hide-metadata toggle,
Metadata-vs-Safetensors conditional, DropZone) is ported as-is.

## 3. What gets copied in (cherry-picked)

**Components** (from DT `components/`):
- `file-tree`, `image-preview`, `metadata-panel`, `metadata-viewer`,
  `metadata-edit-dialog`, `safetensors-panel`, `drop-zone` (+ their direct deps —
  e.g. shadcn `empty`, any small helpers).
- **Not ported:** `thumbnail-viewport` (drawer removed), `navbar`, and DT's UI-kit
  dumps (`animate-ui`, `doras-ui`, `kokonutui`, `optics`, `uitripled`, `glass-*`,
  `chat`).

**Lib:** `parseImageMetadata.ts`, `png-metadata.ts` (+ direct deps).

**Page:** new `frontend/app/dataset-tools/page.tsx` — DT's `page.tsx` with the two deltas.

**Dep:** `react-resizable-panels` (add if the trainer lacks it).

## 4. API routes

Copy the routes this view needs into the trainer, **namespaced** to avoid the
collisions traced in BETA_PLANNING DT-1:

| DT route | Trainer route |
|---|---|
| `/api/fs` | `/api/dataset-tools/fs` |
| `/api/find-file` | `/api/dataset-tools/find-file` |
| `/api/image` | `/api/dataset-tools/image` |
| `/api/thumbnail` | `/api/dataset-tools/thumbnail` |
| `/api/metadata` | `/api/dataset-tools/metadata` |
| `/api/metadata-from-file` | `/api/dataset-tools/metadata-from-file` |
| `/api/metadata-write` | `/api/dataset-tools/metadata-write` |
| `/api/safetensors` | `/api/dataset-tools/safetensors` |

**Required companion change:** add `'/api/dataset-tools'` to `nodeApiPrefixes` at
`frontend/server.js:240`. Without it, `server.js` proxies these to FastAPI:8000
(which has none of them) → 404. This is the one-line whitelist entry (not a proxy
block). All ported components' `fetch` calls get repointed to the namespaced paths.

**Skipped this slice:** `civitai`, `comfyui-nodes`, `rules`, `health`, `settings`
(see §6), and any thumbnail-drawer-only code.

## 5. File-access base

DT's `fs`/`image`/`thumbnail` routes confine access to a configured base folder
(`settings.currentFolder`, passed as a query param). Anchor the base to the trainer's
`PROJECT_ROOT` (cwd-independent, `services.core.validation` / the Node equivalent) so
`datasets/`, `output/`, `pretrained_model/`, `vae/`, and `ComfyUI/` are browsable.
On VastAI this resolves to `/workspace/Ktiseos-Nyx-Trainer`. The user can still
navigate into subfolders; `..` traversal stays blocked.

## 6. Settings

DT tracks `settings.currentFolder` (and a few view prefs) in its own settings system.
Keep this **namespaced and self-contained** — a DT-scoped client store (and/or
`/api/dataset-tools/settings`) — so it never reads/writes the trainer's
`user_settings.json`. No merge with trainer settings in this slice.

## 7. Shared UI dedup

The trainer already has shadcn `components/ui/*`. For each ui primitive a ported
component imports: if the trainer already has it, **use the trainer's** (repoint the
import) and watch for prop drift; only copy in ui primitives the trainer lacks. Goal:
one `components/ui/` set, no duplicates (the same discipline just applied to KNX's own
`src/` cruft).

## 8. Navigation

Add a **"Dataset Tools"** entry under the trainer's existing **Dataset menu** →
`/dataset-tools` (BETA_PLANNING DT-2). No new top-level nav section.

## 9. Out of scope (this slice)

- **Workflow Viewer** — deferred follow-on; it will slot into the right panel as a
  tab beside Metadata (renders the ComfyUI workflow embedded in the selected image,
  not a JSON dump). Pulls in `ComfyUIWorkflowViewer`, `comfyui-node-registry`,
  `comfyui-github-search`, `/api/comfyui-nodes` — none in this slice.
- Thumbnail-drawer component, DT navbar, DT's UI-kit graveyard.
- Merging DT settings into trainer settings; reconciling DT `fs` with the trainer's
  existing `/api/files`.

## 10. Testing

- **API routes:** a smoke test per ported route (namespaced path resolves through
  `server.js` → Next handler, base-folder confinement rejects `..`).
- **Metadata parse:** unit tests for `parseImageMetadata` on a couple of fixture
  images (A1111 + ComfyUI chunks) — this is the load-bearing logic.
- **Layout:** manual — the page renders with the drawer gone, toggle switches
  FileTree list/thumbnail, metadata/safetensors panel shows per file type.

## 11. Rollout

| Step | Deliverable |
|---|---|
| 1 | Copy components + lib + page (deltas applied); add `react-resizable-panels` if needed |
| 2 | Copy + namespace the 8 API routes → `/api/dataset-tools/*`; add the `server.js` whitelist line; repoint fetches |
| 3 | Anchor fs base to `PROJECT_ROOT`; dedupe shared `components/ui/*` |
| 4 | Add the Dataset-menu nav entry → `/dataset-tools` |
| 5 | Tests + manual layout check |

## 12. Follow-on slices (not now)

Workflow Viewer tab · deeper settings/`fs` reconciliation · (already shipped
separately: OKLCH theme picker).
