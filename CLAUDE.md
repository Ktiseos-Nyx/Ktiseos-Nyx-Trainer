# CLAUDE.md — Claude Code Instructions

See **[AGENTS.md](./AGENTS.md)** for the complete project guide — architecture,
critical rules, key components, platform requirements, and code quality
expectations. Everything there applies here too.

This file adds Claude Code-specific guidance on top of that foundation.

---

## Claude Code Behaviour

### Comments and Docstrings

Default to writing **no comments**. Only add one when the WHY is non-obvious:
a hidden constraint, a subtle invariant, or a workaround for a specific bug.
Code that explains itself through good naming doesn't need a comment.

Add or update docstrings **inline** when writing or modifying a function. Don't
batch them into a separate pass — this prevents CodeRabbit from flagging missing
docstrings as a separate concern on the PR. Don't use CodeRabbit's "generate
docstrings" button; write them directly in the code.

### Response Style

Use clear, structured guidance. Prefer short, direct answers over long prose.
When referencing code, include `file_path:line_number` so the user can navigate
directly to it.

### Security

Run `snyk_code_scan` for any new first-party code in a Snyk-supported language.
Fix issues before marking work complete, then re-scan to confirm.

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

## Windows Development

The primary development machine is **Windows 10**. When writing code:

- Use `os.path.join()` or forward slashes — never hardcoded backslashes
- `asyncio.WindowsProactorEventLoopPolicy` is set in `api/main.py` — do not remove
- `start_services_local.bat` is the Windows startup script
- Subprocess env vars `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1` are required when
  spawning Python subprocesses on Windows to avoid `UnicodeEncodeError` on cp1252
- npm and Node commands must work from Windows CMD or PowerShell, not just bash
- Path casing matters for webpack on Windows — NTFS is case-insensitive but webpack
  is not; always use the correct mixed-case path

---

## Documentation Files

- **AGENTS.md** — primary guide (architecture, rules, components)
- **CLAUDE.md** — this file (Claude Code-specific additions)
- **frontend/CLAUDE.md** — frontend-specific detail (Next.js, shadcn/ui, API client)
- **README.md** — user-facing overview
