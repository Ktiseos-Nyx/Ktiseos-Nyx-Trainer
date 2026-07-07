# Optimizer Extension Plan

**Date:** 2026-07-07
**Status:** Draft
**Sources:**
- [Prodigy+Schedule-Free](https://github.com/LoganBooker/prodigy-plus-schedule-free)
- [Personalized-Optimizers](https://github.com/Clybius/Personalized-Optimizers)
- [customized-optimizers](https://github.com/67372a/customized-optimizers)

---

## Context: What We Already Have

Our vendored `trainer/derrian_backend/custom_scheduler/` already ships
**LoraEasyCustomOptimizer** — a massive collection of 67 optimizer files
including CAME, Compass, ADOPT, AdEMAMix, Apollo, D-Adaptation, Galore,
Prodigy, Schedule-Free, SOAP, SPAM, and many more. These are already
available to the Kohya training backend.

The `customized-optimizers` pip package (by the same user) is a curated
subset of 13 optimizers (ABMOG, CAME, FFTDescent, OAGOpt, OCGOpt, SCGOpt,
TALON, etc.) that overlap significantly with what we already have vendored.

**So what's missing?**

---

## 1. Prodigy+Schedule-Free (`prodigy-plus-schedule-free`)

The biggest gap. Combines two popular techniques into one optimizer:

- **Prodigy** — Adaptive LR estimation (no need to tune learning rate)
- **Schedule-Free** — No LR scheduler needed (constant LR, internal interpolation)

**v2.0.0 features:**
- Per-group LR adaptation (train multiple networks at once)
- StableAdamW (soft scaling based on RMS of updates)
- SPEED mode — alternative to Prodigy's numerator/denominator ratio
- Experimental: C-Optim (cautious stepping), Grams, ADOPT, OrthoGrad, FOCUS
- Adam-atan2 support (`eps=None` for scale invariance)
- Fused backward pass support
- Factored second moment (Adafactor-style, lower memory)

**Status:** `pip install prodigy-plus-schedule-free` — pip package, 97 stars,
v2.0.1 released Sep 2025, actively maintained.

**Our current setup:** We have Prodigy and Schedule-Free as *separate*
optimizer choices in the UI, but not the combined optimizer. This means
users can't easily use both together with proper integration.

### Integration Options

**Option A: Add as an optimizer choice in the training UI**

Add `"ProdigyPlusScheduleFree"` to the optimizer dropdown. The Kohya TOML
generator maps it to the correct optimizer name and passes params.

**Effort:** Small — add to optimizer list in config, add params to TOML gen.

**Option B: Install the pip package in installer.py**

`customized-optimizers` (the 67372a package) might serve as a simpler
path — one pip install, 13 optimizers exposed cleanly. But we'd need to
make sure they don't conflict with the vendored versions in
LoraEasyCustomOptimizer.

**Effort:** Trivial — `pip install customized-optimizers` in installer.py.

**Option C: Curate our own optimizer registry**

Rather than pip-installing, cherry-pick optimizer files from these repos
into our vendored `custom_scheduler/LoraEasyCustomOptimizer/`, updating
what's stale. This is what we already do — but some files may be outdated.

**Effort:** Medium — audit 67 existing files, compare with upstream.

### Recommendation

**Phase 1:** Add `ProdigyPlusScheduleFree` as a training optimizer option
(Option A). It's the most impactful addition — users get adaptive LR +
no scheduler in one package.

**Phase 2:** Audit the vendored LoraEasyCustomOptimizer against the sources
above. Many of the 67 files are research prototypes; some may be stale.
Replace/update the ones that have upstream fixes.

---

## 2. Personalized-Optimizers (Clybius)

Research-focused optimizer collection, many already in LoraEasyCustomOptimizer:

| Optimizer | In LoraEasyCustomOptimizer? | Notes |
|-----------|---------------------------|-------|
| OCGOpt | Yes (`ocgopt.py`) | Also in customized-optimizers v2 |
| TALON | Yes (`talon.py`) | Also in customized-optimizers |
| FCompass | Yes (`fcompass.py`) | FAdam + Compass hybrid |
| FishMonger | Yes (`fishmonger.py`) | FIM-based natural gradient |
| FARMSCrop / V2 | Yes (`farmscrop.py`) | Fisher-accelerated RMSProp |
| FMARSCrop variants | Yes (`fmarscrop.py`) | Fisher-accelerated MARS |

The Clybius versions may have more recent updates. Worth comparing
`custom_scheduler/LoraEasyCustomOptimizer/talon.py` against
`Personalized-Optimizers/TALON.py` for drift.

---

## 3. customized-optimizers (67372a)

13 optimizers in a clean pip-installable package. Some of these differ from
what's in LoraEasyCustomOptimizer:

| Optimizer | In LoraEasyCustomOptimizer? | Notes |
|-----------|---------------------------|-------|
| ABMOG | Yes (`abmog.py`) | Same |
| AdamW8bitKahan | Yes (`adam.py` has Kahan variant) | Possibly updated |
| CAME | Yes (`came.py`) | CAME is widely used |
| FFTDescent | Yes (`fftdescent.py`) | Already used |
| OAGOpt | Yes (`oagopt.py`) | Already used |
| OCGOpt | Yes (`ocgopt.py`) | Already used |
| SCGOpt | Yes (`scgopt.py`) | Already used |
| SimplifiedAdEMAMix | Not directly | AdEMAMix is there but simplified variant is new |
| SingState | Yes (`singstate.py`) | Already used |
| SNOO_ASGD | Yes (`snoo_asgd.py`) | Already used |
| TALON | Yes (`talon.py`) | Already used |

---

## Implementation Plan

### File Changes

| File | Change |
|------|--------|
| `services/models/training.py` | Add `ProdigyPlusScheduleFree` to optimizer enum/list |
| `services/trainers/kohya_toml.py` | Map new optimizer name → Kohya params |
| `frontend/lib/api.ts` | Update optimizer options if hardcoded |
| `installer.py` | Optionally `pip install customized-optimizers` |

### Optimizer Config Model

The Prodigy+Schedule-Free optimizer needs these fields in the training config:

```python
class ProdigyPlusScheduleFreeConfig(BaseModel):
    use_schedulefree: bool = True
    use_speed: bool = False
    d0: float = 1e-6
    d_coef: float = 1.0
    d_limiter: bool = True
    prodigy_steps: int = 0          # 0 = never stop adapting
    split_groups: bool = True
    split_groups_mean: bool = False
    factored: bool = True
    use_stableadamw: bool = True
    use_cautious: bool = False
    use_grams: bool = False
    use_adopt: bool = False
    use_orthograd: bool = False
    use_focus: bool = False
    eps: Optional[float] = 1e-8     # None = atan2
```

### UI

The optimizer selector in the training config page already supports:
- AdamW8bit, Adafactor, SGD, D-Adaptation, Prodigy, CAME, etc.

Adding "Prodigy+Schedule-Free" means:
1. New entry in the optimizer dropdown
2. Conditional sub-options panel for the config fields above
3. Best presented with sensible defaults and an "Advanced" toggle for
   the experimental features (C-Optim, Grams, ADOPT, OrthoGrad, FOCUS)

---

## Open Questions

- Should we add `customized-optimizers` as a pip dependency, or keep
  everything vendored in LoraEasyCustomOptimizer? (Dual maintenance problem)
- How stale is LoraEasyCustomOptimizer? Needs an audit against Clybius's
  Personalized-Optimizers and the customized-optimizers package.
- Does Prodigy+Schedule-Free v2.0 work correctly with Kohya's training
  loop? The fused backward pass note says Kohya hard-codes optimizer
  support — may need a patch in our Kohya trainer code.
- Should the Prodigy+Schedule-Free sub-options have presets
  (e.g., "Default", "SPEED mode", "Cautious") to simplify the UI?
