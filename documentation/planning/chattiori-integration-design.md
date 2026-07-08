# Chattiori-Model-Merger Integration Design

**Status:** Draft  
**Priority:** Medium  
**References:** memory #42, `BETA_PLANNING.md` §3.3, `2026-07-07-final-batch-summary.md`  
**License:** Apache 2.0 (MIT-compatible)

---

## 1. Goal

Replace ALL of our current checkpoint merge implementations with Chattiori-Model-Merger as **the** merge engine. This includes:

| Currently | Replaced by Chattiori |
|-----------|----------------------|
| `tools/merge_models.py` (basic weighted average, flaky) | Chattiori WS mode |
| `services/block_weight_merge.py` (our 4-mode clean-room, 339 lines to maintain) | Chattiori's 24+ modes with block weights |
| `custom/anima_merge_lora.py` (Anima LoRA bake) | **No** — Chattiori is checkpoint-only, LoRA bake stays ours |
| No Anima checkpoint-to-checkpoint merge (didn't exist) | Chattiori covers Anima via `detect_arch` + `BLOCKIDAM` |

LoRA merge + LoRA-to-checkpoint bake paths stay unchanged — Chattiori doesn't handle LoRA files.

**What we gain:** 24+ merge modes (vs. 4), cosine structure blending, elemental weights, DARE, Ortho, Sparse, ReBasin, proper metadata, VAE bake-in, Anima support, Flux support, and cross-arch FWM — all for the cost of wiring a subprocess call under Apache 2.0.

---

## 2. What's Wrong Now

| Pain Point | Detail |
|-----------|--------|
| **Basic merge** (`POST /checkpoint/merge`) uses `tools/merge_models.py` | Simple weighted average only. No block weights, no cosine, no DARE. Flaky with different architectures. |
| **Block-weighted merge** (`POST /checkpoint/merge-weighted`) uses `block_weight_merge.py` | Clean-room implementation has only 4 modes (Weight/Add/Triple/Twice). No cosine, no elemental, no DARE/Ortho/Sparse. 339 lines — we're maintaining a mini merger. |
| **No metadata** | Merged checkpoints lose provenance info. No recipe tracking. |
| **No VAE bake** | Must be done as a separate step. |
| **No cross-arch merge** | FWM (Feature Weighted Merge) doesn't exist — can't merge between different base models of same arch. |
| **MG-7: No progress** | Long merges show a spinner with zero feedback. |

---

## 3. What Chattiori Brings

### 3.1 Mode coverage

| Category | Modes | We have? | Chattiori has? |
|----------|-------|----------|----------------|
| Basic blend | WS (Weighted Sum), SIG, GEO, MAX | Weight only | WS, SIG, GEO, MAX |
| Difference-based | AD, sAD, MD, SIM, TD | Add only | AD, sAD, MD, SIM, TD |
| Triple/Tensor | TRS, TS, ST | Triple, Twice | TRS, TS, ST |
| Advanced | DARE, ORTHO, SPRSE, NORM | None | All 4 |
| Structure-aware | CHAN, FREQ, cosine0/1/2 | None | All |
| Cross-arch | FWM (Feature Weighted Merge) | None | Yes |
| Utility | NoIn, RM, SWAP, CLIPXOR, XDARE, COMP | None | All |
| VAE bake | `--vae` | Separate tool | Built-in |
| Pruning | `--prune --keep_ema` | None | Yes |
| Elemental | Per-parameter-type weights | None | Yes |
| Block weights | 19/25-length + auto-convert for XL | 26/20/35 blocks (our own) | Yes + Flux/ZI/Anima |
| ReBasin (weight matching) | Hungarian alignment | None | Yes |

### 3.2 Key features we'd gain immediately

- **Cosine structure modes** (`--cosine0/1/2`) — keep one model's structure, inject details from others. Huge for mixing styles without breaking composition.
- **DARE, Sparse Top-k, Orthogonalized Delta** — proven stochastic merge methods that the community uses for concept mixing.
- **Elemental syntax** — per-parameter-type ratios (attn vs mlp vs conv) without writing block weights.
- **Rich metadata** — SHA256s, merge recipe (`sd_merge_recipe`), per-model provenance (`sd_merge_models`).
- **Flux support** — already handles Flux.1 block mapping.
- **ReBasin** — Hungarian weight matching to align model permutations before merging (reduces "frying").
- **Read-metadata mode** (`RM`) — dump checkpoint metadata without loading the whole model.

### 3.3 What stays ours

- **LoRA merge** (`POST /lora/merge`) — still uses Kohya's `networks/{type}_merge_lora.py`. Chattiori is checkpoint-only.
- **LoRA-to-checkpoint bake** (`POST /lora/merge-to-checkpoint`) — still uses Kohya's `merge_to_sd_model` path + our `custom/anima_merge_lora.py` for the Anima lane. Chattiori doesn't handle LoRA files.
- **Block-weight presets** — our `block_weights.json` + preset picker in the UI. Chattiori's `mbwpresets_master.txt` supplements, doesn't replace.
- **Anima LoRA bake** — `custom/anima_merge_lora.py` stays for the bake-into-checkpoint flow. But **Anima checkpoint-to-checkpoint** merge (which doesn't exist today) goes to Chattiori.

---

## 4. Integration Strategy

### 4.1 Approach: CLI subprocess (same pattern as our Kohya scripts)

```
POST /checkpoint/merge-chattiori  →  subprocess: python merge.py <mode> <args>
```

No Python import, no in-process monkey-patching. Same architecture as:
- `lora_service.merge_lora()` → `networks/sd_merge_lora.py` subprocess
- `lora_service.merge_checkpoint()` → `tools/merge_models.py` subprocess

**Why not in-process?**
1. Chattiori `merge.py` is 2000+ lines with 40+ imports. In-process risks dependency conflicts with our vendored Kohya.
2. CLI subprocess gives us clean stdout/stderr for log streaming and cancellation (kill process).
3. Same pattern we already use — less new infrastructure.

**Where to clone:** `trainer/chattiori/` (alongside `derrian_backend/`). Not a submodule — vendored copy like the rest.

### 4.2 Requirements check

Chattiori's `requirements.txt`: `torch`, `safetensors`, `diffusers`, `lora`, `bitsandbytes`

We already have: `torch`, `safetensors`, `diffusers`, `bitsandbytes`  
We need: `lora` (PyPI package) — should be a simple `pip install lora` addition to `requirements.txt`.

---

## 5. API Design

### 5.1 New endpoint

```
POST /utilities/checkpoint/merge-advanced
```

**Request (`CheckpointAdvancedMergeRequest`):**
```typescript
{
  mode: string,                        // WS | SIG | GEO | MAX | AD | sAD | MD | SIM | TD
                                       // TRS | TS | ST | DARE | ORTHO | SPRSE | NORM
                                       // CHAN | FREQ | SWAP | CLIPXOR | XDARE | FWM
  model_path: string,                  // directory containing models
  model_0: string,                     // filename of model A
  model_1: string,                     // filename of model B
  model_2?: string,                    // filename of model C (required by some modes)
  output: string,                      // output filename (no extension, saved in output/)
  alpha?: number | string,             // float or block-weight string like "0.45" or "0.1,0.2,...,0.5"
  beta?: number | string,              // second ratio (required by some modes)
  device?: "cpu" | "cuda",             // default cpu
  save_safetensors?: boolean,          // default true
  save_half?: boolean,                 // fp16
  cosine0?: boolean,                   // structure from model 0
  cosine1?: boolean,                   // structure from model 1
  cosine2?: boolean,                   // structure from model 2 (requires model_2)
  vae?: string,                        // path to VAE file for bake-in
  prune?: boolean,                     // prune EMA/slim checkpoint
  keep_ema?: boolean,                  // keep EMA only when pruning
  rebasin?: number,                    // ReBasin iterations (0 = skip)
  fine?: string,                       // finetune key pattern
  seed?: number,                       // for stochastic modes (DARE)
  memo?: string,                       // custom metadata note
}
```

**Response:**
```typescript
{
  success: boolean,
  output_path: string,
  file_size_mb: number,
  mode: string,
  metadata: {                           // merge recipe for provenance
    merge_mode: string,
    alpha: string,
    beta?: string,
    cosine_mode?: string,
    models: string[],
    chattiori_version: string,
  }
}
```

### 5.2 Reuse existing endpoints unchanged

| Endpoint | Stays? | Notes |
|----------|--------|-------|
| `POST /checkpoint/merge` | **Deprecate** | Simple weighted merge via `tools/merge_models.py`. Chattiori's WS mode replaces this. Remove after migration. |
| `POST /checkpoint/merge-weighted` | **Remove** | Our 4-mode `block_weight_merge.py`. Chattiori's block-weight support fully supersedes this. |
| `POST /checkpoint/merge-advanced` | **New** | Chattiori-powered. Covers all checkpoint-to-checkpoint merging going forward. |
| `POST /checkpoint/detect-arch` | **Keep** | Still useful for UI. |
| `POST /lora/merge` | **Keep** | LoRA-only, not in Chattiori scope. |
| `POST /lora/merge-to-checkpoint` | **Keep** | LoRA bake (SD/SDXL via Kohya, Anima via our script). Not in Chattiori scope. |

### 5.3 Progress reporting (fixes MG-7)

Chattiori uses `tqdm` internally. We capture subprocess stdout/stderr and pipe progress from tqdm output lines. Same pattern as how we capture Kohya training logs for the WS log viewer.

**Implementation:** Parse tqdm lines (`\r`-based progress) from stderr, emit as job progress updates.

---

## 6. UI Restructure — Dedicated Merge Hub

### 6.1 Problem: One-page tab jungle

The current `/utilities` page has 4 tabs crammed into one page, and the Checkpoint tab has two sub-modes (Basic / Block-Weighted) behind a button toggle. Adding Chattiori as a 5th tab makes it worse — different merge *types* are mixed together, and the page has no room for mode descriptions, preset browsing, or result history.

### 6.2 Solution: Split into `/merge` hub with dedicated pages

Replace the single monolithic page with a **Merge Hub** at its own route, with focused sub-pages for each merge type:

```
/merge                                   ← Hub / landing (overview + recent merges)
├── /merge/checkpoints                   ← Checkpoint-to-checkpoint (Basic + Chattiori Advanced)
├── /merge/loras                         ← LoRA-to-LoRA merging
├── /merge/bake                          ← LoRA → Checkpoint baking
└── /merge/resize                        ← LoRA resizing
```

Each page has:
- Its own URL (linkable, bookmarkable, back-button friendly)
- Room for mode descriptions, parameter docs, and result history
- Consistent layout but tailored to that merge type's controls
- Shared model file picker component across all of them

The `/utilities` page keeps non-merge tools (HF upload, etc.).

### 6.3 Page details

#### `/merge` — Hub / Landing

```
┌──────────────────────────────────────────────────────────┐
│  Merge Tools                                             │
│  Pick the type of merge you want to perform              │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  Checkpoints  │  │   LoRAs      │  │  LoRA → Bake │   │
│  │  A + B = new  │  │ Multi-LoRA   │  │  Bake into   │   │
│  │  model        │  │ merge into   │  │  checkpoint  │   │
│  │               │  │ single LoRA  │  │              │   │
│  │  [25 modes]   │  │  [4 types]   │  │  [3 archs]   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                          │
│  ┌──────────────┐                                        │
│  │  Resize LoRA │                                        │
│  │  Change rank │                                        │
│  │  of a LoRA   │                                        │
│  └──────────────┘                                        │
│                                                          │
│  ── Recent Merges ──                                     │
│  • 2026-07-08 style_mix_v3 — WS + cosine — 1.2 GB        │
│  • 2026-07-07 char_merge — DARE — 2.1 GB                 │
│  • ...                                                    │
└──────────────────────────────────────────────────────────┘
```

#### `/merge/checkpoints` — Checkpoint Merge (replaces current Merge Checkpoints tab)

This is where Chattiori lives. Two sub-modes:

**Basic** — Simple weighted average (existing `tools/merge_models.py`). Model A/B/C picker, ratio sliders, precision. Stays for quick merges.

**Advanced** — Full Chattiori engine with mode picker, block weights, cosine, ReBasin, VAE bake, elemental, all 24 modes.

```
┌──────────────────────────────────────────────────────────────┐
│  ┌──────────────────────────────────────────────────────┐    │
│  │  [Basic]  [Advanced]                                 │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  Mode: [WS ▼]          ← grouped dropdown with category     │
│                           headers: Basic / Diff / Triple /   │
│                           Adv / Structure / Utility          │
│                                                              │
│  Models ────────────────────────────────────────────────     │
│  Model A: [pretrained_model/A.safetensors ▼]                 │
│  Model B: [pretrained_model/B.safetensors ▼]                 │
│  Model C: [ComfyUI/... ▼]  ← shown only for 3-model modes   │
│                                                              │
│  Ratios ────────────────────────────────────────────────     │
│  Alpha: [0.45]  or  [Block Weights...]  ← toggle           │
│  Beta:  [0.20]  ← shown only for modes that need it         │
│                                                              │
│  Options ───────────────────────────────────────────────     │
│  ☐ Cosine structure  [Model 0 ▼]                            │
│  ☐ Bake VAE  [vae_file.safetensors ▼]                       │
│  ☐ Prune (keep EMA only)   ☐ ReBasin [0]                    │
│  ☐ Finetune keys: [comma,separated]                         │
│  Seed: [42]  Precision: [fp32 ▼]  Format: [safetensors ▼]   │
│                                                              │
│  Presets ───────────────────────────────────────────────     │
│  [Load from Chattiori...]  [Load from SuperMerger...]        │
│  [Saved presets...]                                           │
│                                                              │
│  Output: [auto_name]  [Start Merge]                          │
│                                                              │
│  Mode help ─────────────────────────────────────────────     │
│  WS (Weighted Sum): Standard linear interpolation.            │
│  Good for general purpose merging. α=0.45 means 45% B.       │
│  [Show more...]                                               │
└──────────────────────────────────────────────────────────────┘
```

**Mode picker design:**
- Dropdown with `<optgroup>` style categories: Basic, Difference, Triple, Advanced, Structure, Utility
- Each mode has a tooltip on hover explaining what it does
- When mode changes, the UI adapts (show/hide beta, model C, options)

**Block weight editor:**
- Reuses existing `BlockWeightPicker` component but extended for 19/25-length
- Shows block name labels (IN00, IN01, etc.) with sliders
- Preset dropdown loads from both our `block_weights.json` and Chattiori's `mbwpresets_master.txt`

#### `/merge/loras` — LoRA Merge (moved from current tab)

Same functionality as current Merge LoRAs tab. Dedicated page gives room for:
- Per-LoRA ratio cards with drag-to-reorder
- Block-weight per LoRA (if we ever re-enable, per Dusk's call)
- Result history

#### `/merge/bake` — LoRA → Checkpoint (moved from current tab)

Same as current LoRA → Checkpoint tab, but with:
- Dedicated space for the 3-arch toggle (SD/SDXL/Anima)
- TE path picker for Anima
- Better output preview

#### `/merge/resize` — LoRA Resize (moved from current tab)

Simple dedicated page. Minimal change.

### 6.4 Backward compat

- `/utilities` keeps working — redirects to `/merge` for relevant actions, keeps HF upload and other non-merge tools
- A transition banner on `/utilities` points to the new merge hub
- No breaking URL changes

**Layout:**

```
┌──────────────────────────────────────────────┐
│ Merge LoRAs | Merge Checkpoints | LoRA→Ckpt | ■ Advanced Merge │
└──────────────────────────────────────────────┘

  Mode: [WS ▼]  ← dropdown of all 24 modes

  Model Directory: [pretrained_model/ ▼]  ← trainer + ComfyUI dirs
  Model A: [dropdown: pick safetensors]
  Model B: [dropdown: pick safetensors]
  Model C: [dropdown: pick safetensors]  ← shown only when mode needs 3 models

  ── Ratios ──
  Alpha: [0.45]  or  [Block Weights...]  ← toggle between simple float and block-weight editor
  Beta:  [0.20]  ← shown only when mode needs beta

  ── Options ──
  □ Cosine structure  [Model 0 ▼]  ← dropdown: Model 0 / Model 1 / Model 2
  □ Bake VAE: [VAE file picker...]
  □ Prune (keep EMA only)
  □ ReBasin: [0] iterations
  □ Finetune keys: [comma separated...]
  □ Seed: [42]  ← for DARE etc.

  Precision: [fp32 ▼]  |  Format: [safetensors ▼]

  ── Presets ──
  [Styles] [Characters] [Concepts]  ← preset combo selectors
  [Load from Chattiori presets...]  ← loads from mbwpresets_master.txt

  Output: [auto-generated name]

  [Start Merge]
```

### 6.2 Preset integration

- Our existing `block_weights.json` presets remain for the Checkpoint Merge tab's Block-Weighted mode.
- Chattiori's `mbwpresets_master.txt` used in the Advanced tab as an additional preset source.
- The Advanced tab's block-weight editor reuses the existing `BlockWeightPicker` component pattern but feeds Chattiori's 19/25-length format.

### 6.3 Mode descriptions

Each mode needs a tooltip/help card explaining what it does (copied from Chattiori's README). Modes grouped by category:

```
Basic:        WS (Weighted Sum), SIG (Sigmoid), GEO (Geometric), MAX (Max)
Difference:   AD, sAD, MD, SIM, TD
Triple:       TRS, ST, TS
Advanced:     DARE, ORTHO, SPRSE, NORM
Structure:    CHAN, FREQ, (cosine toggles)
Utility:      SWAP, CLIPXOR, XDARE, FWM, COMP
```

---

## 7. Attribution & License

| Component | License | How we use it |
|-----------|---------|---------------|
| Chattiori-Model-Merger (`Faildes/Chattiori-Model-Merger`) | **Apache 2.0** | Vendored at `trainer/chattiori/`. Apache 2.0 is MIT-compatible — we can redistribute modified copies. |
| Preset arrays (`mbwpresets_master.txt`) | Data (facts) | Load at runtime as preset definitions. Already referenced as "safe lifts as data" in `BETA_PLANNING.md`. |
| Our additions/modifications | MIT | Any new code we write for the integration (API, UI, service) is MIT as part of this repo. |

**Attribution file:** `ATTRIBUTIONS.md` already exists. Add entry:
> "Chattiori-Model-Merger by Faildes (Apache 2.0) — used as the checkpoint merge engine in the Advanced Merge tab. Source: https://github.com/Faildes/Chattiori-Model-Merger"

---

## 8. Build Order

### Phase 1 — Backend + Foundation (can be done offline, no GPU needed)

1. **Clone** Chattiori to `trainer/chattiori/`
2. **Add** `lora` to Python dependencies (`requirements.txt` / `installer.py`)
3. **Vendored backend smoke test** — run `merge.py RM` on any safetensors to verify basic loading
4. **Create** `services/merge_advanced_service.py` — subprocess wrapper with tqdm parsing, progress tracking, timeout
5. **Create** API endpoint `POST /merge/checkpoints/advanced` + Pydantic models
6. **Add** `utilitiesAPI.mergeAdvanced()` to `api.ts`

### Phase 2 — UI Restructure (frontend-only, no GPU)

1. **Create** `frontend/app/merge/page.tsx` — Merge Hub landing page
2. **Create** `frontend/app/merge/checkpoints/page.tsx` — Checkpoint merge (Basic + Advanced tabs)
3. **Create** `frontend/app/merge/loras/page.tsx` — LoRA merge (moved from utilities tab)
4. **Create** `frontend/app/merge/bake/page.tsx` — LoRA → Checkpoint (moved from utilities tab)
5. **Create** `frontend/app/merge/resize/page.tsx` — LoRA resize (moved from utilities tab)
6. **Build** the Advanced mode picker (grouped dropdown + adaptive form)
7. **Build** mode help cards (descriptions from Chattiori README)
8. **Add** preset loading from `mbwpresets_master.txt`
9. **Add** VAE file picker integration (reuse existing model dir listing)
10. **Add** transition banner on `/utilities` pointing to `/merge`
11. **Remove** `block_weight_merge.py` + `POST /checkpoint/merge-weighted` (after verifying zero usage)
12. **Keep** old `/merge/checkpoints/basic` endpoint for backward compat

### Phase 3 — GPU verification (needs Dusk/GPU)

1. **Test** all 24 modes against known model pairs (A/B/C test sets)
2. **Cross-check** WS mode output against our existing `merge_models.py` to verify match
3. **Test** cosine modes with real style images to verify structure preservation
4. **Test** ReBasin on fried model pairs
5. **Audit** block-weight mapping against Chattiori's `BLOCKID`/`BLOCKIDXLL`

### Phase 4 — Polish

1. Add saved preset system (named presets per user)
2. Add recent merge history to `/merge` hub
3. Mode descriptions tooltips refinement
4. Deprecate old utilities tab page

---

## 9. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Chattiori has deps that conflict with our vendored Kohya | Low | CLI subprocess means separate Python process; no import-level conflict. Can even run in a different venv if needed. |
| Chattiori's in-place tensor mutation surprises | Medium | The `merge.py` script mutates model0's theta in-place. Our wrapper should clone inputs or verify idempotency. |
| tqdm parsing is fragile for progress | Low | If tqdm output changes, we fall back to start/end-only logging (what we have now). |
| Some modes produce unusable checkpoints | Medium | That's inherent to experimental merge modes, not a bug in Chattiori. Users accept this risk. |
| ReBasin adds significant time | Low | It's opt-in (`--rebasin 0` by default). |
| `lora` PyPI package version conflicts | Low | Pinned in `requirements.txt`. Tested on install. |

---

## 10. Future Considerations

- **CLIP-XOR** and **XDARE** modes are interesting for CLIP manipulation — could eventually get their own simplified UI.
- **FWM (Feature Weighted Merge)** enables merging models with different backbones (e.g., SD1.5 into SDXL). This is cutting-edge and worth spotlighting in the UI once verified.
- Chattiori's `--cfg_sens` and `--sat_boost` flags for CFG sensitivity scaling could be exposed as expert options.
