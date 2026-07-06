# Dataset-Tools Browse View Port — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port Dataset-Tools' Browse view into the trainer as an in-app page at `/dataset-tools` (under the Dataset navbar menu), replicating DT's layout minus its navbar and the redundant bottom thumbnail drawer.

**Architecture:** Cherry-pick DT's Browse components + parsing libs + the 8 API routes they need into namespaced trainer locations. Routes go under `/api/dataset-tools/*` and get one `server.js` whitelist entry so `server.js` hands them to Next instead of proxying to FastAPI. The Workflow Viewer is stubbed out of `metadata-panel` for this slice.

**Tech Stack:** Next.js 16 (App Router), React 19, shadcn/ui, Tailwind v4, `react-resizable-panels`. Source app: `C:\Users\dusk\Development\Dataset-Tools` (`knx-dataset-tools`, same stack).

## Global Constraints

- **shadcn/ui only** — no raw HTML form elements; reuse the trainer's `components/ui/*`, only copy in primitives it lacks.
- **Namespacing** — all ported code lives under a `dataset-tools` namespace: `components/dataset-tools/`, `lib/dataset-tools/`, `app/api/dataset-tools/`. Ported types → `types/dataset-tools/`.
- **API routing** — every ported route MUST be reachable: add `'/api/dataset-tools'` to `nodeApiPrefixes` in `frontend/server.js` (~line 240). Ported `fetch` calls use `/api/dataset-tools/*`.
- **fs base** — file access confines to `PROJECT_ROOT` (Node-side, file-anchored, cwd-independent), never process CWD. Block `..` traversal.
- **Cross-platform** — `path` module / forward slashes; preserve path casing.
- **Verify each task** — `cd frontend && ./node_modules/.bin/tsc --noEmit` must stay at 0 errors after each task that touches TS.
- **Reference, don't reinvent** — copy component/route bodies from the DT source path; this plan defines *where they go* and *what to adapt*, not their full source.

**Component/route closure (traced from DT source):**

| Ported file (DT) | Deps that come with it | API routes it calls |
|---|---|---|
| `components/file-tree` | ui: context-menu, dropdown-menu, empty, tooltip · `use-settings` · types: fs, metadata | `fs`, `thumbnail` |
| `components/image-preview` | (react, lucide only) | — |
| `components/metadata-panel` | `metadata-edit-dialog` · ui: empty, tooltip · `use-settings` · types: metadata, rules · **WF: `ComfyUIWorkflowViewer`, `comfyui-node-registry`** ← STUB | `metadata`(via lib), **`comfyui-nodes`, `rules`** ← STUB |
| `components/metadata-viewer` | `lib/parseImageMetadata` | — |
| `components/metadata-edit-dialog` | ui: button, checkbox, dialog, textarea · sonner | `metadata-write` |
| `components/safetensors-panel` | `lib/utils` · types: safetensors | — |
| `components/drop-zone` | (react, lucide only) | — |
| `lib/parseImageMetadata` | (none) | — |
| `lib/png-metadata` | (none) | `metadata` |

---

### Task 1: Ported types + parsing libs (foundation, no UI)

**Files:**
- Create: `frontend/types/dataset-tools/{fs,metadata,rules,safetensors}.ts` (copy from DT `types/`)
- Create: `frontend/lib/dataset-tools/parseImageMetadata.ts`, `frontend/lib/dataset-tools/png-metadata.ts` (copy from DT `lib/`)
- Test: `frontend/__tests__/dataset-tools/parseImageMetadata.test.ts`

**Interfaces:**
- Produces: `parseImageMetadata(raw: string | Record<string, unknown>)` and the `types/dataset-tools/*` type exports consumed by later tasks. Keep DT's exported names verbatim.

- [ ] **Step 1: Copy the type files and libs** from DT into the namespaced paths above. In `png-metadata.ts`, change any `fetch('/api/metadata...')` → `fetch('/api/dataset-tools/metadata...')`. `parseImageMetadata.ts` has no imports to change.

- [ ] **Step 2: Write a failing parse test** using two fixture strings (an A1111 `parameters` chunk and a ComfyUI `prompt` JSON chunk) copied from a real image's metadata:

```ts
import { parseImageMetadata } from '@/lib/dataset-tools/parseImageMetadata';
test('detects A1111 parameters chunk', () => {
  const r = parseImageMetadata('masterpiece, 1girl\nNegative prompt: bad\nSteps: 20, Sampler: Euler a');
  expect(r.format).toBe('A1111'); // adjust key to DT's actual return shape
});
test('detects ComfyUI prompt JSON', () => {
  const r = parseImageMetadata(JSON.stringify({ '3': { class_type: 'KSampler' } }));
  expect(r.format).toBe('ComfyUI');
});
```

- [ ] **Step 3: Run it, confirm it fails** (or reveals DT's real return shape): `cd frontend && ./node_modules/.bin/tsc --noEmit && npx vitest run __tests__/dataset-tools/parseImageMetadata.test.ts` — Expected: FAIL / shape mismatch. Adjust assertions to DT's actual return keys, then it passes (the parser itself is ported, not rewritten).

- [ ] **Step 4: Commit**

```bash
git add frontend/types/dataset-tools frontend/lib/dataset-tools frontend/__tests__/dataset-tools
git commit -m "feat(dataset-tools): port metadata parsing libs + types"
```

---

### Task 2: Namespaced API routes + server.js whitelist

**Files:**
- Create: `frontend/app/api/dataset-tools/{fs,find-file,image,thumbnail,metadata,metadata-from-file,metadata-write,safetensors}/route.ts` (copy from DT `app/api/`)
- Modify: `frontend/server.js` (~line 240 — add `'/api/dataset-tools'` to `nodeApiPrefixes`)
- Create: `frontend/lib/dataset-tools/base-path.ts` (fs-base resolver)
- Test: `frontend/__tests__/dataset-tools/base-path.test.ts`

**Interfaces:**
- Produces: `resolveBase(): string` → the trainer `PROJECT_ROOT`; `assertWithinBase(target: string): string` → throws on `..`/out-of-base, returns the safe absolute path. All 8 routes use these.

- [ ] **Step 1: Write `base-path.ts` failing test**

```ts
import { assertWithinBase } from '@/lib/dataset-tools/base-path';
test('rejects traversal outside base', () => {
  expect(() => assertWithinBase('../../etc/passwd')).toThrow();
});
test('accepts a path inside base', () => {
  expect(assertWithinBase('datasets/foo.png')).toContain('datasets');
});
```

- [ ] **Step 2: Implement `base-path.ts`** — resolve `PROJECT_ROOT` file-anchored (mirror how the trainer's Node services anchor: `path.resolve(process.cwd(), '..')` is WRONG if cwd is project root; use a source-file-relative anchor or an explicit env like the trainer already uses). Confine with `path.resolve` + prefix check (blocks `..`). Run test → PASS.

- [ ] **Step 3: Copy the 8 route files** into `app/api/dataset-tools/*`. In each, replace DT's base-folder handling with `assertWithinBase(...)` from Task 2, and leave the rest of the logic intact (`sharp` thumbnail pipeline, metadata read/write, fs listing).

- [ ] **Step 4: Add the whitelist line** in `frontend/server.js`:

```js
const nodeApiPrefixes = ['/api/jobs', '/api/files', '/api/captions', '/api/settings', '/api/dataset', '/api/dataset-tools', '/api/config', '/api/civitai', '/api/utilities', '/api/debug'];
```

- [ ] **Step 5: Smoke-test one route** — start the app, `curl 'http://localhost:3000/api/dataset-tools/fs?path=datasets'` → returns a JSON listing (not a 404 / FastAPI error). Confirms the whitelist + route wiring.

- [ ] **Step 6: Commit**

```bash
git add frontend/app/api/dataset-tools frontend/server.js frontend/lib/dataset-tools/base-path.ts frontend/__tests__/dataset-tools/base-path.test.ts
git commit -m "feat(dataset-tools): namespaced fs/image/metadata/safetensors routes + server.js whitelist"
```

---

### Task 3: DT settings hook + shared-ui reconciliation

**Files:**
- Create: `frontend/hooks/use-dt-settings.ts` (copy DT `hooks/use-settings.ts`, renamed to avoid any clash)
- Create (only if trainer lacks them): `frontend/components/ui/{context-menu,dropdown-menu,empty,tooltip,checkbox}.tsx` via shadcn
- Modify: ported components' imports → `use-dt-settings`

**Interfaces:**
- Produces: `useDtSettings()` returning at least `{ currentFolder, setCurrentFolder }` (match DT's shape). Self-contained (localStorage or `/api/dataset-tools/settings`) — never touches the trainer's `user_settings.json`.

- [ ] **Step 1: Copy `use-settings.ts` → `use-dt-settings.ts`**, keeping its state self-contained; if it hit `/api/settings`, repoint to `/api/dataset-tools/settings` (add that route too, copied from DT `app/api/settings`).
- [ ] **Step 2: Inventory shared ui** — for each ui primitive in the closure table, check `frontend/components/ui/`. Present already: `button`, `dialog`, `textarea` (verify). Add missing via `npx shadcn@latest add <name>` (likely `context-menu`, `dropdown-menu`, `empty`, `tooltip`, `checkbox`).
- [ ] **Step 3: Verify** `cd frontend && ./node_modules/.bin/tsc --noEmit` → 0 errors (hook + ui primitives resolve).
- [ ] **Step 4: Commit**

```bash
git add frontend/hooks/use-dt-settings.ts frontend/components/ui
git commit -m "feat(dataset-tools): DT settings hook (namespaced) + shared ui primitives"
```

---

### Task 4: Leaf components (clean copies)

**Files:**
- Create: `frontend/components/dataset-tools/{image-preview,drop-zone,safetensors-panel,metadata-viewer,metadata-edit-dialog}.tsx` (copy from DT)

**Interfaces:**
- Produces: these components' default/named exports, consumed by `metadata-panel`, `file-tree`, and the page. Keep DT export names.

- [ ] **Step 1: Copy the five files** into `components/dataset-tools/`. Repoint imports: `@/lib/parseImageMetadata` → `@/lib/dataset-tools/parseImageMetadata`; `@/lib/utils` → `@/lib/utils` (trainer's, unchanged); `@/types/*` → `@/types/dataset-tools/*`; `/api/metadata-write` → `/api/dataset-tools/metadata-write` (in `metadata-edit-dialog`). ui imports stay `@/components/ui/*` (trainer's).
- [ ] **Step 2: Verify** `./node_modules/.bin/tsc --noEmit` → 0 errors.
- [ ] **Step 3: Commit**

```bash
git add frontend/components/dataset-tools
git commit -m "feat(dataset-tools): port leaf components (preview, dropzone, safetensors, metadata-viewer/edit)"
```

---

### Task 5: metadata-panel — port with Workflow Viewer STUBBED

**Files:**
- Create: `frontend/components/dataset-tools/metadata-panel.tsx` (copy from DT, adapted)

**Interfaces:**
- Consumes: `metadata-viewer`, `metadata-edit-dialog`, `use-dt-settings`, `types/dataset-tools/metadata`.
- Produces: `MetadataPanel` with the same props DT's page passes (`metadata`, `isLoading`, `filePath`, `baseFolder`, `onRefresh`).

- [ ] **Step 1: Copy `metadata-panel.tsx`**, then remove the Workflow Viewer coupling: delete the `import ... ComfyUIWorkflowViewer` line, delete the `@/lib/comfyui-node-registry` import, remove the workflow tab/trigger from the tab list, and remove the `fetch('/api/comfyui-nodes')` / `fetch('/api/rules')` calls (and any state they feed). Leave a `// TODO(dataset-tools slice 2): restore Workflow Viewer tab` marker where the tab was.
- [ ] **Step 2: Repoint remaining imports** to the namespaced paths (`@/components/dataset-tools/metadata-edit-dialog`, `@/hooks/use-dt-settings`, `@/types/dataset-tools/metadata`).
- [ ] **Step 3: Verify** `./node_modules/.bin/tsc --noEmit` → 0 errors, and no residual reference to `comfyui-nodes`/`rules`/`ComfyUIWorkflowViewer`:

```bash
grep -rn "comfyui-nodes\|ComfyUIWorkflowViewer\|/api/rules" frontend/components/dataset-tools/metadata-panel.tsx || echo "clean"
```

- [ ] **Step 4: Commit**

```bash
git add frontend/components/dataset-tools/metadata-panel.tsx
git commit -m "feat(dataset-tools): port metadata-panel with Workflow Viewer tab stubbed (slice 2)"
```

---

### Task 6: file-tree (copy + repoint)

**Files:**
- Create: `frontend/components/dataset-tools/file-tree.tsx` (copy from DT)

**Interfaces:**
- Consumes: `/api/dataset-tools/fs`, `/api/dataset-tools/thumbnail`, `use-dt-settings`, `types/dataset-tools/{fs,metadata}`, ui: context-menu/dropdown-menu/empty/tooltip.
- Produces: `FileTree` with DT's props (`onFileSelect`, `onDirExpand`, `selectedFile`, `viewMode`, `refreshKey`).

- [ ] **Step 1: Copy `file-tree.tsx`**; repoint `/api/fs`→`/api/dataset-tools/fs`, `/api/thumbnail`→`/api/dataset-tools/thumbnail`, `@/hooks/use-settings`→`@/hooks/use-dt-settings`, `@/types/*`→`@/types/dataset-tools/*`. ui imports unchanged.
- [ ] **Step 2: Verify** `./node_modules/.bin/tsc --noEmit` → 0 errors.
- [ ] **Step 3: Commit**

```bash
git add frontend/components/dataset-tools/file-tree.tsx
git commit -m "feat(dataset-tools): port file-tree (list/thumbnail viewMode)"
```

---

### Task 7: Browse page (deltas applied) + nav entry

**Files:**
- Create: `frontend/app/dataset-tools/page.tsx` (DT `app/page.tsx` minus the two deltas)
- Modify: the trainer navbar (Dataset menu) — add "Dataset Tools" → `/dataset-tools`

**Interfaces:**
- Consumes: all Task 4–6 components + `use-dt-settings`.

- [ ] **Step 1: Copy DT `app/page.tsx` → `frontend/app/dataset-tools/page.tsx`.** Apply DELTA 1 — delete the `id="thumbnails"` `Panel` + `ThumbnailViewport` and the enclosing vertical `PanelGroup`; the center `Panel` renders `ImagePreview` directly. Apply DELTA 2 — do not render DT's `Navbar` (page renders inside the trainer chrome). Repoint all component imports to `@/components/dataset-tools/*` and hook to `@/hooks/use-dt-settings`; repoint the inline `/api/image?...` src to `/api/dataset-tools/image?...`.
- [ ] **Step 2: Add nav entry** — find the trainer navbar's Dataset menu (`grep -rn "Dataset" frontend/components/blocks/navigation/`), add an item linking to `/dataset-tools` labelled "Dataset Tools".
- [ ] **Step 3: Verify build** `cd frontend && ./node_modules/.bin/tsc --noEmit` → 0 errors, and `npm run build` succeeds.
- [ ] **Step 4: Manual smoke** — run the app, open Dataset menu → Dataset Tools: file tree lists `PROJECT_ROOT`; List/Thumbnail toggle switches FileTree rendering; selecting an image shows preview + parsed metadata; selecting a `.safetensors` shows the safetensors panel; there is NO bottom thumbnail drawer.
- [ ] **Step 5: Commit**

```bash
git add frontend/app/dataset-tools/page.tsx frontend/components/blocks/navigation
git commit -m "feat(dataset-tools): Browse page (drawer removed, in-trainer nav) + Dataset menu entry"
```

---

## Self-Review

- **Spec coverage:** layout deltas (Task 7), cherry-picked components (Tasks 4–6), libs/types (Task 1), 8 namespaced routes + whitelist (Task 2), fs base = PROJECT_ROOT (Task 2), settings self-contained (Task 3), ui dedup (Task 3), nav entry (Task 7), WF viewer deferred (Task 5 stub). ✓
- **Deviation from spec:** `metadata-panel` is edited (WF stub) rather than copied verbatim — forced by its coupling; recorded in Task 5. The spec's "skip comfyui-nodes/rules" is satisfied via the stub.
- **Open at execution:** confirm DT's actual `parseImageMetadata` return shape (Task 1 Step 3) and which ui primitives the trainer already has (Task 3 Step 2) — both resolved against real files, not guessed.

## Out of scope (later slices)
Workflow Viewer tab (un-stub Task 5 + port `ComfyUIWorkflowViewer`/`comfyui-node-registry`/`comfyui-github-search` + `/api/comfyui-nodes`,`/api/rules`); deeper DT-settings/`fs` reconciliation with the trainer's `/api/files`.
