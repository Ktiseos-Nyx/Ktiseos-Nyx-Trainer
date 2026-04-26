# GEMINI.md — Gemini Code Assistant Instructions

See **[AGENTS.md](./AGENTS.md)** for the complete project guide — architecture,
critical rules, key components, platform requirements, and code quality
expectations. Everything there applies here too.

This file adds Gemini-specific notes on top of that foundation.

---

## Gemini-Specific Reminders

### Do Not Modify Git Remotes

Gemini tools have previously attempted to reconfigure remote URLs. The
repository is already correctly configured. Never run any `git remote` commands.

### Do Not Change 0.0.0.0 Bindings

Gemini's security heuristics may flag `host="0.0.0.0"` in uvicorn or FastAPI
startup calls. **Do not change these.** The binding is intentional for
VastAI and RunPod cloud deployment. See AGENTS.md for the full explanation.

### Outdated Path References

The following paths appear in older Gemini conversation history but **do not
exist** in the current codebase. Do not create or reference them:

| Outdated | Correct |
|---------|---------|
| `core/managers.py` | `services/training_service.py` |
| `core/dataset_manager.py` | `services/dataset_service.py` |
| `core/kohya_training_manager.py` | `services/trainers/kohya.py` |
| `core/utilities_manager.py` | `services/lora_service.py` |
| `shared_managers.py` | does not exist |
| `api/routers/` | `api/routes/` |
| `frontend/src/` | `frontend/app/` |
| `CLAUDE_DEVELOPMENT_RULES.md` | does not exist |
| `CLAUDE_PROJECT_STATUS.md` | does not exist |

Always read AGENTS.md → Key Components for accurate file locations.

### Framework Versions

- Next.js **15** (not 14) with React **19**
- Tailwind CSS **v4**
- Python 3.10+, Node.js 20+
