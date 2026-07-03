# Additive Multi-Source Download System — Design

**Status:** Draft for review (KNX-internal + external review by Arc En Ciel dev)
**Date:** 2026-07-03
**Scope:** v1 = pluggable source-adapter framework + Arc En Ciel adapter. Civitai left as-is.

---

## 1. Context & Goal

Ktiseos-Nyx-Trainer (KNX) is a LoRA training tool: a FastAPI backend + Next.js
frontend that runs both **locally** (end-user machines) and on **remote rented GPU
boxes** (e.g. VastAI, accessed through a tunnel). Users download base models, LoRAs,
VAEs, and embeddings into either the trainer's own folders (for training) or the
bundled ComfyUI's folders (for generation).

**Today:**
- KNX has its own **Civitai** downloader. It works but is stale/fragile in the
  *search + URL-resolution* layer (Civitai's gated, quirky API). The underlying
  download **engine** is solid.
- **LoRA Manager (LM)** — the ComfyUI extension KNX already ships — handles Civitai
  well (batch URLs, rich metadata, confirmation). LM is the intended long-term owner
  of Civitai acquisition.
- **LM does not support Arc En Ciel** (`arcenciel.io`), a smaller community model site.

**Goal:** an *additive* download system — a small framework where new model **sources**
can be added over time behind a common interface. The first new source is **Arc En
Ciel**, precisely because LM won't do it. Over time, Civitai acquisition is handed to
LM and KNX owns the sources LM won't.

**Non-goals:**
- No forking of LoRA Manager (decided: LM stays as-is, unforked).
- No rebuilding what LM already does well.
- v1 does **not** touch the existing Civitai code.

---

## 2. Constraints & Principles

1. **Native / owned.** No dependency on forking a third-party project.
2. **Must work on remote instances.** Downloads run **server-side on the box**, so the
   user's browser can be anywhere (behind a tunnel). This is load-bearing — it's the
   reason the old Civitai link worked on remote boxes, and it must be preserved.
3. **Cross-platform** (Windows/macOS/Linux). Use `os.path`/`pathlib`, inject
   `PYTHONIOENCODING=utf-8`/`PYTHONUTF8=1` for any subprocess, no hardcoded separators.
4. **Contain each source's quirks in its own adapter.** A bug or API change in one
   source cannot destabilize another.
5. **Quality bar = LM.** Batch URLs, real metadata, SHA-256 confirmation.
6. **Honesty about unknowns.** Arc has **no public API docs**; its *download auth* is
   unverified. The design isolates that unknown into a single, swappable step plus a
   dedicated discovery phase — it is not papered over.

---

## 3. Architecture — two independent axes

The system separates **what** you download from (sources) from **how** a download is
triggered (triggers). They are independent layers.

```
┌─ Trigger layer (Axis 2) ────────────────────────────┐
│  v1:   In-app pull   (search in KNX UI → /download)  │  ← works LOCAL + REMOTE
│  later: Desktop Link (arcenciel.io push → worker)    │  ← mostly local; remote via
└──────────────────────┬──────────────────────────────┘     outbound websocket
                       │  both call ↓
┌─ Source adapters (Axis 1) ──────────────────────────┐
│  SourceAdapter interface + registry                  │
│   ├─ CivitaiAdapter    (NOT built in v1; legacy path │
│   │                     stays; future LM handoff)    │
│   ├─ ArcEnCielAdapter  (v1: search confident,        │
│   │                     download pending Phase 0)    │
│   └─ (future: HuggingFace, direct URL, …)            │
└──────────────────────┬──────────────────────────────┘
                       │  produces a DownloadSpec
┌─ Download engine (already exists) ──────────────────┐
│  services/model_service.py: fallback chain           │
│  (hf_hub → aria2c → wget → requests), resume,        │
│  SHA-256 verify, filename preservation, dest routing │
└──────────────────────────────────────────────────────┘
```

**v1 scope:** the `SourceAdapter` framework + `ArcEnCielAdapter` + in-app pull UI,
feeding the existing engine. The desktop Link (Axis 2) and Civitai adapter are **out of
scope for v1** but the seams are reserved for them.

---

## 4. Components

### 4.1 `SourceAdapter` interface

A source adapter knows how to *browse, describe, and resolve a download URL* for one
model site. It does **not** perform the download itself — it produces a `DownloadSpec`
the shared engine consumes.

```python
class SourceAdapter(Protocol):
    name: str                      # "arcenciel", "civitai", …
    requires_api_key: bool

    def search(self, query: SearchQuery) -> SearchResult: ...
    def get_model(self, model_id: str) -> ModelDetail: ...
    def get_versions(self, model_id: str) -> list[ModelVersion]: ...
    def resolve_download(self, version: ModelVersion,
                         api_key: str | None) -> DownloadSpec: ...
    def base_model_classes(self) -> list[str]: ...   # for search filters
```

**Shared data models** (source-agnostic; each adapter maps its site's JSON into these):

- `SearchQuery { term, sort, page, limit, base_model?, model_type? }`
- `ModelSummary { source, model_id, title, type, base_model, cover_url, nsfw }`
- `ModelDetail { …summary…, description, versions[] }`
- `ModelVersion { version_id, name, base_model, model_type, files[], sha256?, cover_url }`
- `DownloadSpec { url, filename, sha256?, size?, headers?, dest_type }`
  - `dest_type` is a normalized enum (`checkpoint | diffusion_model | lora | vae |
    text_encoder | embedding`) used for destination routing (§4.4).

A **registry** maps `name → adapter`. Routes and UI iterate the registry so adding a
source is one new class + one registration line.

### 4.2 `ArcEnCielAdapter`

Verified from two community reference clients (Anzhc's A1111 extension; FallenIncursio's
ComfyUI node) — treated as *leads, not gospel* (see §6).

- **Base URL:** `https://arcenciel.io/api`
- **Search (confirmed unauthenticated GET):**
  `GET /models/search?search=&sort=&page=&limit=&baseModel=&modelType=`
- **Classes / base models:** `GET /models/classes`
- **Detail:** `GET /models/{id}`
- **Versions:** `GET /models/{id}/versions` → carries file info + (per Anzhc) an
  `externalDownloadUrl` and/or `filePath`, and a SHA-256.
- **Gallery (optional, for UI previews):** `GET /models/{id}/gallery`
- **Download URL resolution (per Anzhc):**
  `version.externalDownloadUrl` if present, else
  `https://arcenciel.io/api/models/{id}/versions/{version_id}/download`
- **Auth for download:** **UNKNOWN — see Phase 0 (§5) and Questions (§6).** Implemented
  behind a single `_attach_auth(request, api_key)` method so the resolved mechanism is a
  one-place change.
- **Assets base (thumbnails):** `https://arcenciel.io/uploads/{path}`

### 4.3 Download engine (reuse, no changes expected)

`services/model_service.py::download_model_or_vae(DownloadConfig)` already:
- takes a URL + optional `api_token`, downloads **server-side** (remote-safe),
- runs a fallback chain (hf_hub → aria2c → wget → requests) with resume,
- preserves filename and verifies size; SHA-256 verification is added/confirmed here.

The adapter's `DownloadSpec` maps directly onto `DownloadConfig`. If Arc's auth turns
out to need a **header** (not a URL token), we add a small `headers` pass-through to the
engine's request methods — the one anticipated engine change, gated on Phase 0.

### 4.4 Destination routing

Reuse the existing pattern from the Civitai endpoint (`api/routes/civitai.py:262`):
a request carries a `destination` (`training` vs `comfyui`) and, for ComfyUI, a subfolder.
The adapter's normalized `dest_type` maps to a concrete directory:

| `dest_type` | training dest | comfyui dest |
|---|---|---|
| `checkpoint` | `pretrained_model/` | `ComfyUI/models/checkpoints/` |
| `diffusion_model` | `pretrained_model/` | `ComfyUI/models/diffusion_models/` |
| `lora` | `output/` | `ComfyUI/models/loras/` |
| `vae` | `vae/` | `ComfyUI/models/vae/` |
| `text_encoder` | `pretrained_model/` | `ComfyUI/models/text_encoders/` |
| `embedding` | `pretrained_model/` | `ComfyUI/models/embeddings/` |

ComfyUI path resolution uses `get_comfyui_models_path()` (recently hardened to prefer the
bundled `{project_root}/ComfyUI/models`).

### 4.5 API routes / process boundary

New endpoints live in FastAPI (which already owns model downloads):
`GET /api/sources`, `GET /api/sources/{name}/search`,
`GET /api/sources/{name}/models/{id}`, `POST /api/sources/{name}/download`.

**No `server.js` whitelist change needed:** the Node server only routes a fixed set of
`/api/*` prefixes to itself and sends everything else to FastAPI:8000. `/api/sources/*`
falls through to FastAPI automatically.

### 4.6 Frontend UI

A source-agnostic **Browse & Download** surface (shadcn/ui only — `Select`, `Input`,
`Button`, `Card`), modeled on the existing Civitai browse tab:

- A **source selector** (`Civitai | Arc En Ciel`) driving the same results grid.
- Search box + filters (base model, type) populated from the adapter.
- Result cards → version picker → **destination choice** (training vs ComfyUI folder).
- **Batch selection** (queue multiple, mirroring LM's batch-URL strength).
- Live download progress from the job manager.
- **Settings:** an "Arc En Ciel API key" field alongside the existing Civitai key.

---

## 5. Phase 0 — Arc API discovery (the honest unknown)

**Why:** Arc has no docs; the two reference extensions use *different* auth models from
*different* eras, and neither has been re-verified. We will not spec a download mechanism
we can't confirm.

**What:** a small throwaway probe (run once, with the maintainer's approved account/key):
1. `GET /models/search?...` — confirm open access + response shape.
2. `GET /models/{id}/versions` — confirm file metadata: filename, size, SHA-256 field,
   `externalDownloadUrl` vs `/…/download`, model_type/base-model taxonomy.
3. Hit the **download** URL three ways and record what happens:
   - no auth,
   - API key as a header (`Authorization: Bearer …` and/or `x-api-key`),
   - API key as a query param.
   Record: `200` (direct file), `302` (redirect to a signed CDN URL), `401/403`
   (needs session), and any `Set-Cookie`/`Location`/rate-limit headers.

**Output:** pins `ArcEnCielAdapter.resolve_download` + `_attach_auth`, and answers "is
server-side pull viable, or is this login/Link-Key-only?" (which decides whether the
desktop Link becomes *required* rather than optional for Arc).

---

## 6. Open Questions for Arc En Ciel / the Arc dev

Grouped and prioritized. **[BLOCKER]** = needed before v1 download can be built.
**[LATER]** = informs the future desktop Link, not v1.

### Auth & download (blockers)
1. **[BLOCKER]** Is there an **API key** usable from a server (no live browser session)?
   Where is it generated (account settings?), and what form does it take?
2. **[BLOCKER]** How is that key presented on a request — `Authorization: Bearer <key>`,
   an `x-api-key` header, a query param, or something else?
3. **[BLOCKER]** Does `GET /api/models/{id}/versions/{vid}/download` return the file
   directly, or **302-redirect** to a signed/expiring CDN URL? If signed: does the
   signed URL need auth too, or is it pre-authorized (fetchable by any client)?
4. **[BLOCKER]** Is `externalDownloadUrl` present for *all* downloadable versions, or
   only some (e.g. models hosted off-site)? When it's present, does it need auth?
5. **[BLOCKER]** Do **gated / early-access / NSFW** models use a *different* download
   path or permission than public ones?

### Metadata & taxonomy (blockers for correct routing)
6. **[BLOCKER]** In `/models/{id}/versions`, which fields carry: file **name**, file
   **size**, **SHA-256**, and file **format** (safetensors/ckpt/pt)?
7. **[BLOCKER]** What are the possible `modelType` values (checkpoint, LoRA, VAE,
   embedding, DiT/UNet, …)? We map these to download destinations.
8. What `baseModel` / class values does `/models/classes` return (Illustrious, NoobAI,
   Pony, SDXL, SD1.5, Flux, …)? Used for search filters.

### Operational (good-to-know)
9. Are there **rate limits** or a required **User-Agent** for API/download requests?
10. Is there a **batch/bulk** endpoint, or should we iterate per-model?
11. Any **API usage terms / attribution** we should honor for a first-party integration?

### Desktop Link — future (later)
12. **[LATER]** For a desktop "Connect" integration, is the **Link Key (`lk_...`)** the
    current recommended mechanism (vs the legacy API key)? What's the handshake?
13. **[LATER]** The Link worker connects **outbound** to `link.arcenciel.io` over
    websocket — is that correct, and would it work for a headless remote box (no browser
    on the box) as long as it can reach Arc?
14. **[LATER]** Is the download command over the Link channel a pre-signed URL the worker
    then fetches (as it appears in FallenIncursio's client), or something else?

---

## 7. Testing

- **Adapter unit tests** (mock Arc HTTP responses, GPU-free): search parsing, version
  parsing, `resolve_download` for both `externalDownloadUrl` and `/download` cases,
  `dest_type` → directory mapping.
- **Engine**: already robust; add a SHA-256-mismatch test if not present.
- **Phase 0 probe** is a manual, one-off script — not part of CI.

---

## 8. Rollout

| Phase | Deliverable |
|---|---|
| **0** | Arc API discovery probe → answers §6 blockers, pins auth/resolution |
| **1** | `SourceAdapter` interface + registry + `ArcEnCielAdapter` (search + resolve) + FastAPI routes + wire to engine |
| **2** | Frontend Browse & Download UI (source selector, batch, destination) + Arc API-key setting |
| **3 (later)** | Civitai acquisition handed to LM; deprecate/remove KNX's Civitai downloader |
| **4 (later)** | Desktop Link bridge (Axis 2) — outbound-WS worker, Link Key |

---

## 9. Out of scope (v1)

- Forking LoRA Manager (permanently off the table).
- Touching or "fixing" the existing Civitai downloader.
- The desktop Link bridge (designed-for, not built).
- Any auth mechanism assumed without Phase-0 confirmation.
