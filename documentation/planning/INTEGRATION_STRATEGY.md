# ComfyUI Integration Strategy — Fork / Workflow / Native

**Scope of this doc:** ComfyUI- and frontend-specific integration *only* — which
existing nodes/extensions we fork, which capabilities we express as workflow
templates, and where the line to "keep native" falls.

**Out of scope (lives elsewhere, do not poach):**
- **Training** — native owned core. AGENTS.md / the trainer itself.
- **Model merging + LoRA→checkpoint bake** — native, already specced in
  `BETA_PLANNING.md` (MG-section, MG-1→MG-14), built on our vendored kohya.
- Other long-term BETA_PLANNING lanes (dataset tools, etc.) — they keep their
  own lane.

This is the *tactical* layer under the **"orchestration not ownership"** scope
decision: training is the owned moat; gen/tooling is mostly orchestrated. This
doc records the case-by-case calls for the ComfyUI side, which accrete **over
time** — nothing here is a commitment to build now.

---

## The heuristic — native / workflow / fork

> Reuse the ecosystem; don't re-roll wheels. But **reuse has a default order:
> orchestrate before you own, and fork only to fix.**

For each ComfyUI/frontend capability, in order of preference:

1. **It already exists and works as a node/extension** → **build a
   workflow/template around it. No fork.** The default for ComfyUI-side
   capabilities (background removal, upscale, detailers, audio…). Orchestrate,
   don't own.
2. **Composable from nodes but no ready workflow** → **author a new
   workflow/template** — starting from a *proven* community workflow, not a blank
   canvas.
3. **An existing extension is *almost* right but broken / missing a piece** →
   **fork it, fix it** — the *exception*, justified only by a real gap (e.g. LoRA
   Manager's metadata cache). Keep the fork minimal: backend patch ≫ UI reskin.
4. **Native owned core** → **build/keep our own version** (training; merge/bake —
   sd-scripts plumbing we already have). Vendor + **attribute** any ComfyUI node
   code we copy.

### Which surface — ComfyUI vs native React/Next?

A second axis, for *where a feature lives*:
- **Interactive / canvas / direct-manipulation** (hand-masking, image editing,
  layout) → leans **native React/Next**. ComfyUI's graph is awkward for fluid,
  click-and-drag work.
- **Pipeline transform** (background removal, upscale, detail/restore, audio gen)
  → leans **ComfyUI workflow**. "A node in a graph" is its natural shape.

### Adding a ComfyUI capability — the path

1. Find a **proven workflow** for it (don't author from scratch).
2. Understand the wiring — `dump-workflow.js` dumps the *real* graph; **ask** if
   node ownership/wiring is unclear, don't guess (`cnr_id` lies). For any node we
   haven't used before, check what it *actually does* (its repo/docs) before
   trusting it — we know certain nodes, not the whole library.
3. **Template it** — freeze the graph, inject the params we expose.
4. Optional **thin React surface** to drive it.
5. Fork a node **only** if a needed piece is broken/missing.

Not EZPZ — but a logically-sound, repeatable path. Mistakes happen; the method
holds.

---

## Candidate inventory

Accretes over time. Verdict = `Fork` / `Workflow` / `Native` / `TBD`.

| Capability | Verdict | Rationale | Cost / Risk | Status |
|---|---|---|---|---|
| **LoRA Manager** (`willmiao/ComfyUI-Lora-Manager`, GPL-3.0) | **Fork** | Not a node — a standalone React app at `/loras` with its own Civitai pipeline, downloads, metadata DB, recipes. Fork to (a) fix the metadata cache bug, (b) merge Civitai downloading into the Ecosystem UI. | Active upstream (moves fast — updated hours apart). Backend-only patch merges clean ~forever; **UI reflow = perpetual merge war**. Clone-at-install ⇒ fork-by-URL, not in-tree patch. | **Deferred / scoping.** Not today. |
| **Background removal (rembg etc.)** | **Workflow** | Pipeline-transform shape — mature nodes already exist (BRIA RMBG / InspyreNet / rembg families). No fork; template around an existing node. | Low. Verify the current best node when picked up. | Idea / exploring |
| **Image editing** | **TBD → likely native** | Interactive-canvas axis → leans React/Next. ComfyUI has inpaint + edit-models, but fluid click-drag editing fights the graph paradigm. | Native canvas is real frontend work. | Idea / exploring |

### LoRA Manager — detail

- **The cache bug (load-bearing):** `Save Image (LoraManager)` (node id 65 in our
  SDXL templates) embeds LoRA recipe metadata into the PNG, but reads it from a
  cache that never busts between queued gens — so image #2+ inherit image #1's
  recipe. Corrupts provenance silently across a batch. Known + unfixed upstream
  ([#615](https://github.com/willmiao/ComfyUI-Lora-Manager/issues/615) asks for
  *automatic* cache refresh; also #730, #754). Maintainer's only answer is manual
  "hit Refresh," which can't survive a "queue 50 and walk away" run.
- **Interim workaround (no code):** rebuild the LoRA Manager cache right before
  each batch.
- **Fix shape (when picked up):** invalidate/rebuild the metadata cache per
  execution (or key Save Image metadata on the live LoRA stack, not the cached
  library).
- **Two separable decisions — don't let the cheap one smuggle the expensive one:**
  1. *Cache fix* — small backend change, clean merges. Cheap.
  2. *Civitai-download UX into Ecosystem* — heavier. **Prefer a thin Ecosystem-side UI that
     drives LoRA Manager's download API over reflowing its `/loras` app** (avoids
     the merge war; fits "orchestration not ownership"). Reflowing the live React
     app is the costliest form of ownership.
- **Second known bug — DiT misplacement (from 2026-06-29):** LM dumps *everything*
  into the `checkpoints` category, but DiT/Anima weights need `diffusion_models/`
  (where `UNETLoader` looks). A download-focused fork inherits this unless it also
  fixes **arch-aware placement** — which the scope direction says the *trainer*
  owns (DiT→`diffusion_models`, all-in-one→`checkpoints`, LoRA→output). This is
  why 6/29 concluded "don't fork **Batchlinks**" (the external tool) for downloads;
  the same caution applies to leaning on LM for placement.
- **We use:** `Lora Loader`, `Checkpoint Loader`, `Prompt`, `Save Image`
  (all `(LoraManager)`); its library symlinks into `ComfyUI/models`, which
  `api/routes/utilities.py:73` already special-cases.

---

## Fork-cost ground rules

- **Cost scales with how much UI you touch.** Backend-only patch → clean merges
  against fast upstreams. UI reflow → merge war; treat as a separate, heavier
  decision.
- **Clone-at-install nodes** (provisioned via `installer.py`, `git pull
  --ff-only`) → fork **by changing the clone URL**, not an in-tree patch (a patch
  breaks the fast-forward pull). This is the opposite of vendored deps
  (sd-scripts/LyCORIS), which carry an in-tree patch-set.
- **License:** ComfyUI and most nodes are GPL-3.0. A fork stays GPL-3.0 with
  source public (it already is). The Ecosystem trainer stays **MIT** — it's a separate
  program talking to ComfyUI over HTTP (aggregation, not a derivative work).
  Attribute any node code we copy/vendor in `ATTRIBUTIONS.md`.
- **Prefer "drive the API" over "reskin the app"** wherever a forked extension
  exposes endpoints.

---

## Open questions

- **LoRA Manager purpose split:** is bulk Civitai acquisition aimed at *gen
  inputs* (LoRAs/checkpoints to generate with — LoRA Manager fits) or *training
  inputs* (base models + datasets into the trainer — LoRA Manager's gen-library
  destination is the wrong place)? This gates how far the fork goes.
- **Placement knot (from 6/29):** "download THROUGH LM for metadata" vs LM
  mis-placing DiT means for DiT the download layer must deliberately choose —
  *bypass LM* (lose metadata) or *download-via-LM-then-correct-placement*. A
  Civitai-download fork has to pick one.
- **Downloader consolidation:** we currently have only a basic model downloader
  (our own "batchlinks" is specced, not built — and note 6/29's "trainer owns
  arch-aware placement" leans toward *building it ours*, not delegating placement
  to LM). Does the basic downloader fold into the LM Civitai path, or stay a
  separate trainer-side downloader? Depends on the purpose + placement above.
