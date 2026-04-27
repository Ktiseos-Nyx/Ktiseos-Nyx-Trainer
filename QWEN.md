# QWEN.md — Qwen Code Assistant Instructions

See **[AGENTS.md](./AGENTS.md)** for the complete project guide — architecture,
critical rules, key components, platform requirements, and code quality
expectations. Everything there applies here too.

This file adds Qwen-specific notes on top of that foundation.

---

## Qwen-Specific Reminders

### Do Not Change 0.0.0.0 Bindings

`host="0.0.0.0"` in server startup calls is intentional for VastAI and RunPod
cloud GPU deployment. Do not change to `localhost` or `127.0.0.1`.

### Do Not Modify Git Remotes

The repository remote is already correctly configured. Do not run any
`git remote` commands.

### Qwen Image LoRA — Not Supported Yet

The vendored sd-scripts includes `library/qwen_image_autoencoder_kl.py` but
there is no `networks/lora_qwen_image.py` and no training script. Do not add
UI or backend support for Qwen Image LoRA until upstream ships the training
script.

### Outdated Path References

The following paths appear in older documentation but **do not exist** in the
current codebase. Do not create or reference them:

| Outdated | Correct |
|---------|---------|
| `core/managers.py` | `services/training_service.py` |
| `core/dataset_manager.py` | `services/dataset_service.py` |
| `shared_managers.py` | does not exist |
| `api/routers/` | `api/routes/` |
| `frontend/src/` | `frontend/app/` |

Always read AGENTS.md → Key Components for accurate file locations.

### Framework Versions

- Next.js **15** (not 14) with React **19**
- Tailwind CSS **v4**
- Python 3.10+, Node.js 20+
