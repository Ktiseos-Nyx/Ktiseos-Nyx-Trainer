# Logging UI — WandB + TensorBoard Config

**Date:** 2026-05-20  
**Branch:** `feat/wandb-tensorboard-logging-ui`  
**Issues:** LT-1 (wandb_key env var), UI-1 (LoggingCard)

---

## Problem

The training config schema already has all the right fields (`log_with`, `logging_dir`, `log_prefix`, `log_tracker_name`, `wandb_run_name`, `wandb_key`) and the TOML generator already writes them — but there is no UI to set them, and `WANDB_API_KEY` is never injected into the Kohya subprocess env. Users cannot configure logging at all from the trainer.

---

## Scope

**In:** UI fields for logging config, WANDB_API_KEY env var injection.  
**Out:** Graph visualisation, TensorBoard process management, `log_tracker_config` (advanced TOML override).

TensorBoard and WandB are both external — TB is bundled on VastAI and launched separately, WandB is a cloud service. The trainer's job is only to configure Kohya correctly, not to launch or embed either tool.

---

## Design

### 1. Backend — `services/trainers/kohya.py`

In the env setup block (alongside `PYTHONPATH`, `PYTHONIOENCODING`):

```python
if self.config.wandb_key:
    env["WANDB_API_KEY"] = self.config.wandb_key
```

One line. No other backend changes needed — TOML generation already handles all logging args.

### 2. Frontend — `LoggingCard.tsx`

New file: `frontend/components/training/cards/LoggingCard.tsx`

**Fields:**

| Field | Type | Shown when | Notes |
|-------|------|-----------|-------|
| `log_with` | Select | Always | None / TensorBoard / WandB / Both |
| `logging_dir` | Input | TB or Both | Path to write event files |
| `log_prefix` | Input | TB or Both | Optional prefix for timestamped folder |
| `wandb_key` | Password Input | WandB or Both | Used as WANDB_API_KEY env var |
| `log_tracker_name` | Input | WandB or Both | WandB project name (optional) |
| `wandb_run_name` | Input | WandB or Both | Individual run name (optional) |

All fields map directly to existing Zod schema fields — no schema changes.

**Conditional visibility:** derive `showTensorboard` and `showWandB` booleans from the `log_with` watch value, render sections accordingly.

### 3. Tab registration — `frontend/components/training/tabs/index.tsx`

Import `LoggingCard` and render it after `SavingCard`. Same `form` prop pattern as every other card.

---

## Data flow

```
LoggingCard UI
  → useTrainingForm (existing hook, existing fields)
  → TrainingConfig (existing Pydantic model)
  → kohya_toml.py (already writes log_with, logging_dir, log_prefix,
                    log_tracker_name, wandb_run_name)
  → kohya.py env setup (NEW: WANDB_API_KEY from wandb_key)
  → Kohya subprocess
```

---

## Files to change

| File | Change |
|------|--------|
| `services/trainers/kohya.py` | Add WANDB_API_KEY env var injection |
| `frontend/components/training/cards/LoggingCard.tsx` | New file |
| `frontend/components/training/tabs/index.tsx` | Import + render LoggingCard |

No changes to: schema, validation, TOML generator, API routes, defaults, or presets.
