# AGENTS.md — Project Guide for AI Coding Agents

This is the primary reference for any AI agent working in this repository.
Agent-specific files (`CLAUDE.md`, `GEMINI.md`, `QWEN.md`) point here and add
per-agent notes on top.

---

## CRITICAL RULES — Read These First

### This is a WEB UI Project (NOT Jupyter Notebooks)

This project uses **FastAPI + Next.js** for the interface, not Jupyter notebooks.

- **NO** `.ipynb` files, notebook cells, or ipywidgets — ever
- **YES** FastAPI backend (Python) + Next.js frontend (React/TypeScript)
- Access via web browser at `http://localhost:3000`
- If you see references to "notebooks" or "Jupyter" in existing code, they are outdated

### Vendored Backend — No Submodules

`trainer/derrian_backend/` is committed directly into the repository.

- **DO** treat it as regular source code
- **DO NOT** run `git submodule` commands — there are no submodules
- **DO NOT** add git submodules to this repository
- **DO NOT** modify vendored backend files unless explicitly asked

### Do Not Change Server Bindings

Do not change `0.0.0.0` to `localhost` or `127.0.0.1`:

```python
uvicorn.run(app, host="0.0.0.0", port=8000)  # correct — leave as-is
```

This application deploys to VastAI and RunPod cloud GPU instances. `0.0.0.0`
is required for external access. Security is handled by the platform firewall
and Cloudflare proxy. Do not change this even if a security scanner flags it.

### Do Not Modify Git Remotes

Never run commands that change remote configuration:

```bash
# Never run:
git remote set-url origin ...
git remote add ...
git remote remove ...
```

### Do Not Create Audit Files

Do not create markdown audit files (`API_AUDIT.md`, `ENDPOINT_AUDIT.md`, etc.)
unless the user specifically asks for one. Report findings directly in the
conversation instead.

---

## Repository Overview

A **web-based** LoRA training environment for training LoRA models for Stable
Diffusion image generation. Uses a FastAPI backend and Next.js frontend,
designed to run cross-platform.

**Architecture**: FastAPI (Python) + Next.js 15 (React/TypeScript) + Kohya SS

**Deployment targets** (all equally supported):
- **Windows local** — primary development environment; code must work here first
- **VastAI** — Linux cloud GPU instances with auto-provisioning and supervisor
- **RunPod** — Linux cloud GPU instances via `provision_runpod.sh`
- **Training requirement** — NVIDIA GPU with CUDA 12.1+

**Not supported**: macOS. Do not write Mac-specific code or instructions.

---

## Key Components

### Backend — `api/`

FastAPI route handlers:

| File | Purpose |
|------|---------|
| `api/main.py` | FastAPI app entry point |
| `api/routes/dataset.py` | Dataset upload and management |
| `api/routes/training.py` | Training configuration and execution |
| `api/routes/models.py` | Model downloads (HuggingFace / Civitai) |
| `api/routes/utilities.py` | LoRA resizing and HuggingFace uploads |
| `api/routes/config.py` | Configuration management |
| `api/routes/files.py` | File operations and browsing |
| `api/routes/settings.py` | Application settings |
| `api/routes/civitai.py` | Civitai integration |
| `api/routes/debug.py` | Debug and diagnostics |

### Service Layer — `services/`

Business logic. Route handlers call services; services call the training
backend. Never call Kohya scripts directly from routes.

| File / Dir | Purpose |
|-----------|---------|
| `training_service.py` | Training job orchestration |
| `tagging_service.py` | WD14 image tagging |
| `captioning_service.py` | BLIP / GIT captioning |
| `dataset_service.py` | Dataset processing and validation |
| `caption_service.py` | Caption file management |
| `model_service.py` | Model download and management |
| `lora_service.py` | LoRA utilities |
| `websocket.py` | WebSocket handlers for real-time logs |
| `jobs/` | Job management system |
| `trainers/` | Training backend integration (Kohya SS) |
| `models/` | Pydantic data models and schemas |
| `core/` | Core utilities and path validation |

### Frontend — `frontend/`

Next.js 15 with React 19, TypeScript, and Tailwind CSS v4.

| Dir | Purpose |
|-----|---------|
| `frontend/app/` | Next.js App Router pages |
| `frontend/components/` | React components |
| `frontend/lib/` | API client (`api.ts`) and utilities |
| `frontend/hooks/` | Custom React hooks |

All backend communication goes through `frontend/lib/api.ts`. Do not fetch the
backend directly from components.

### Training Backend — `trainer/derrian_backend/`

Vendored Kohya SS distribution (committed directly, not a submodule):

| Dir | Purpose |
|-----|---------|
| `sd_scripts/` | Kohya SS training scripts |
| `lycoris/` | LyCORIS library |
| `custom_scheduler/` | Custom optimizers (CAME, Compass, LPFAdamW, etc.) |

### Startup Scripts

| Script | Platform | Purpose |
|--------|---------|---------|
| `install.sh` / `install.bat` | Linux / Windows | Install dependencies |
| `installer.py` | Both | Python dependency installer |
| `start_services_local.sh` | Linux | Start locally |
| `start_services_local.bat` | Windows | Start locally |
| `start_services_vastai.sh` | VastAI | Supervisor-managed startup |
| `provision_runpod.sh` | RunPod | Direct port binding startup |
| `vastai_setup.sh` | VastAI | One-time provisioning |

---

## Development Commands

### Installation

```bash
# Linux
chmod +x ./install.sh && ./install.sh

# Windows
install.bat

# Or directly
python installer.py
```

### Start the Application

**Linux:**
```bash
chmod +x ./start_services_local.sh
./start_services_local.sh
```

**Windows:**
```bat
start_services_local.bat
```

Access:
- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

### Run Tests

```bash
pip install -r requirements_dev.txt
pytest tests/test_api_plumbing.py -v
```

Tests are GPU-free and mock all ML dependencies. `tests/conftest.py` stubs
missing production packages so tests run without the full venv.

---

## Platform and Path Rules

- **Never hardcode paths** — use `sys.executable` and `os.path.join()`
- **Primary dev OS is Windows 10** — do not write Unix-only solutions
- **Deploy target is Linux** (VastAI / RunPod) — code must work on both
- **Do not target macOS** — not a supported platform
- Use `asyncio.create_subprocess_exec` for async subprocesses (not `subprocess.Popen` with shell)
- `api/main.py` sets `asyncio.WindowsProactorEventLoopPolicy()` on Windows — do not remove

---

## Configuration System

Training uses two TOML files generated by the backend API:

- `trainer/runtime_store/dataset.toml` — dataset config (paths, resolution, batch size)
- `trainer/runtime_store/config.toml` — training hyperparameters

These are auto-generated from web UI form inputs. Check git history before
modifying TOML generation logic — mistakes here affect real training runs.

---

## Frontend Rules

### Use shadcn/ui Components — Mandatory

Never use raw HTML form elements. This project uses shadcn/ui for all UI:

| Avoid | Use instead |
|-------|------------|
| `<select>` | `<Select>` from `@/components/ui/select` |
| `<input>` | `<Input>` from `@/components/ui/input` |
| `<button>` | `<Button>` from `@/components/ui/button` |
| `<textarea>` | `<Textarea>` from `@/components/ui/textarea` |

### TypeScript and Patterns

- TypeScript everywhere — no plain `.js` files in `frontend/`
- Use the `@/` path alias, not relative paths
- Mark components that use hooks or browser APIs with `'use client'`
- Prefer server components when no client-side features are needed
- Accessibility is mandatory — semantic HTML, keyboard navigation, ARIA attributes

---

## Code Quality Rules

- **Complete functions** — never leave methods incomplete during refactoring
- **Preserve all functionality** — UI changes are fine; business logic must stay intact
- **Reference old code** when refactoring — compare with working versions before changing
- **Ask before removing** — if unsure about a method's purpose, ask rather than delete
- **Refactor incrementally** — test each step; do not do massive changes at once
- **Compare method counts** before and after large refactors to catch missing functions
- **Add docstrings inline** — when writing or modifying a function, update its docstring then and there

---

## What NOT to Do

- Add Jupyter references to any file
- Run `git submodule` commands
- Modify git remotes
- Change `0.0.0.0` server bindings
- Hardcode absolute paths
- Write macOS-only or Unix-only code
- Edit `trainer/derrian_backend/` without explicit instruction
- Add platform-specific npm packages directly (causes `EBADPLATFORM` on other OSes)
- Create markdown audit files unless explicitly asked

---

## Supported Model Types

- Stable Diffusion 1.5 (LoRA, LoCon, LoHa, LoKR)
- SDXL (LoRA, LoCon, LoHa, LoKR)
- Flux (experimental)
- SD 3.5 (experimental)

**Not yet supported**: Qwen Image LoRA — the vendored sd-scripts has the
autoencoder but no training script. Do not add UI for it until upstream ships
the training script.

---

## Current Status

- Backend refactored to use Kohya's library system with strategy pattern
- Frontend ~99% migrated to Next.js 15 + React 19 routes
- Core training functionality works; in beta / bug-squashing phase
- Flux and SD3 should work based on Kohya backend; needs real-world testing
