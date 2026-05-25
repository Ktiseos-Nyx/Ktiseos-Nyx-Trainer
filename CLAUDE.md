# CLAUDE.md — Claude Code Instructions

See **[AGENTS.md](./AGENTS.md)** for the complete project guide — architecture,
critical rules, key components, platform requirements, and code quality
expectations. Everything there applies here too.

This file adds Claude Code-specific guidance on top of that foundation.

---

## Bleeding Edge — Non-Standard Parameters Are Intentional

This project is a **bleeding edge training tool** used by experienced LoRA trainers who know exactly what they're doing. Parameters that look "wrong" by conventional standards are often intentional.

**Do not:**
- Add artificial min/max/step constraints to training parameter fields (LR, alpha, dim, etc.) based on what "typical" training looks like
- Reject or warn on LR values, optimizer args, or network settings just because they fall outside common tutorials
- "Fix" preset values from community experts (Novowels, Jelosus2, etc.) — treat them as authoritative
- Add validation that enforces conventional wisdom (e.g. TENC LR must be lower than UNet LR — sometimes it isn't)

**Do:**
- Trust that the user and community preset authors understand the implications
- Use `step="any"` on numeric training fields so any precision is valid
- Only validate for hard errors (missing required paths, type mismatches, structurally invalid configs)

If a PR review agent flags a training parameter as "non-standard" or "too high/low" — ignore it. The user operates outside standard ranges by design.

---

## Ask, Don't Assume — Especially for ComfyUI / Workflow Questions

When something about a ComfyUI workflow is unclear (node names, which package a node comes from, how nodes are wired), **ask before making changes**.

- The user can check ComfyUI while it's running and provide node **names** (even if not node IDs)
- The workflow JSON is human-readable — check it first before guessing
- Speculative edits followed by reversions waste far more time than one clarifying question
- Never assume node ownership from JSON metadata alone (`cnr_id` can be wrong/stale)

**The rule:** if confused, ask. One question → correct fix. Guessing → chain of wrong edits → backlog.

---

## Claude Code Behaviour

### Preset Management Rule

Please keep all training presets out of the Next.js/TypeScript codebase. Instead, manage them exclusively through the JSON files in the presets/ folder. This keeps configuration centralized, simplifies version tracking, and avoids the inconsistencies that come with localStorage or hardcoded values.

### Python Optional → Frontend Guard Rule

Any time a Python field is typed `Optional[X]` (i.e. can be `null` in JSON), the corresponding frontend guards **must** use `!= null` (loose inequality), not `!== undefined`. JSON never produces `undefined` — only `null` — so `!== undefined` always passes on a null value and any method call (`.toFixed()`, `.toExponential()`, etc.) will throw at runtime.

**Checklist when adding or changing an `Optional` field on the Python side:**
1. Find the `api.ts` transformer that maps it into the frontend object.
2. Find any render guards in components that check before displaying it.
3. Find any state-merge guards (e.g. spread patterns) that use it.
4. Change all three from `!== undefined` → `!= null`.

This was learned from a `lr` / `loss` null crash in `TrainingMonitor.tsx` (PR b8c6c3d).

### Comments and Docstrings

Prefer self-documenting code through clear naming. Avoid inline comments unless explaining a non-obvious rationale (edge cases, constraints, or workarounds). 

Write and update docstrings inline during development. Never defer them to a separate pass or rely on CodeRabbit's auto-generation. Keeping docstrings in sync with code changes prevents review friction and ensures documentation is part of the implementation, not an afterthought.

### Response Style

Use clear, structured guidance. Prefer short, direct answers over long prose.
When referencing code, include `file_path:line_number` so the user can navigate
directly to it.

### Security & Vulnerability Management

- **Secure Defaults:** Validate/sanitize inputs, enforce least-privilege permissions, never hardcode secrets or credentials, and avoid permissive configurations (e.g., open CORS, debug mode in production).
- **Dependency Hygiene:** Pin versions in lockfiles, audit third-party packages regularly, and replace unmaintained or flagged dependencies promptly.
- **AI Generation Guardrails:** Proactively avoid known vulnerable patterns, suggest secure alternatives, and flag any security-sensitive changes (auth, file I/O, network calls, env vars) during code generation.

---

## Frontend — shadcn/ui (Mandatory)

Never use raw HTML form elements. This project uses shadcn/ui components:

| Avoid | Use instead |
|-------|------------|
| `<select>` | `<Select>` from `@/components/ui/select` |
| `<input>` | `<Input>` from `@/components/ui/input` |
| `<button>` | `<Button>` from `@/components/ui/button` |
| `<textarea>` | `<Textarea>` from `@/components/ui/textarea` |

See `components/training/fields/FormFields.tsx` for examples of correct usage.

---

## Cross-Platform & Environment Conventions
**Primary dev environment:** Windows 10 | **Target runtime:** Cross-platform (Windows/macOS/Linux)

All code must remain cross-platform compatible. When writing or modifying files, enforce:

- **Path handling:** Use `os.path.join()` or forward slashes (`/`). Never hardcode `\`. Always preserve exact casing in paths (NTFS is case-insensitive, but webpack and Unix tooling are not).
- **Python event loop:** `api/main.py` configures `asyncio.WindowsProactorEventLoopPolicy`. Do not remove or conditionally override it.
- **Subprocess encoding:** When spawning Python subprocesses, always inject `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1` into the environment. This prevents Windows `cp1252` errors and is safe across all OSes.
- **Node/npm CLI:** All commands must execute natively in Windows CMD/PowerShell. Avoid bash-specific syntax, Unix path assumptions, or WSL dependencies.
- **Local startup:** `start_services_local.bat` is the Windows launcher. Keep it functional, but ensure any shared scripts or configs it calls remain cross-platform safe.

---

## Documentation Files

- **AGENTS.md** — primary guide (architecture, rules, components)
- **CLAUDE.md** — this file (Claude Code-specific additions)
- **frontend/CLAUDE.md** — frontend-specific detail (Next.js, shadcn/ui, API client)
- **README.md** — user-facing overview
