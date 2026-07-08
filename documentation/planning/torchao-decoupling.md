# TorchAO Import Decoupling

## Problem

`torchao` was pulled into memory on **every** training-validation request and
**every** `import LoraEasyCustomOptimizer`, even when the user selected a
standard optimizer (AdamW, SGD) or a non-AO custom optimizer (CAME, Compass).

This caused:
- Unnecessary memory pressure from torchao's C++ extension registration
  (`torch.ops.load_library`) and eager quantization/dtypes module loading
- Risk of import-order side effects from torchao's `TorchAOBaseTensor`
  metaclass/tensor subclass registration contaminating non-AO training runs
- Merge/bake scripts inheriting the same venv and getting torchao'd too

## Root Cause

Two linked issues:

### 1. `adam.py` module-level import (primary)

`trainer/derrian_backend/custom_scheduler/LoraEasyCustomOptimizer/adam.py:3`:
```python
from .low_bit_optim.adam import AdamW8bit, AdamW4bit, AdamWFp8
```

This fires at **module parse time** — any `import` from `adam.py`, for any
symbol, triggers the cascade:
```
adam.py → low_bit_optim/__init__.py → low_bit_optim/adam.py
  → subclass_{4bit,8bit,fp8}.py → from torchao.utils import TorchAOBaseTensor
```

### 2. `AdamW8bitKahan` collocation

`AdamW8bitKahan` (a bitsandbytes-only optimizer, no torchao dependency) lived
in `adam.py` alongside the 3 AO classes. Importing `AdamW8bitKahan` at module
level in `__init__.py` — which everyone does via `from LoraEasyCustomOptimizer
import OPTIMIZERS` — fired the entire torchao chain.

The "lazy import" in `validation.py` was cosmetic: it delayed the import from
process-start to first `validate()` call, but `__init__.py`'s eager imports
(of everything including adam.py) meant it fired unconditionally.

## Fix

### A. Split `AdamW8bitKahan` → `adam_kahan.py`

Created `adam_kahan.py` containing only:
- `_stochastic_round_bf16` helper
- `AdamW8bitKahan` class (inherits `bitsandbytes.optim.AdamW8bit`, pure bnb)

Now `adam.py` is **never imported at module-load time** — it only loads
lazily when `_LazyAO._resolve()` fires (i.e., when someone actually selects
AdamW8bitAO/4bitAO/fp8AO in training).

### B. Lazy-AO proxy in `__init__.py`

Replaced the eager import:
```python
# OLD — fires torchao at module-import time
from LoraEasyCustomOptimizer.adam import AdamW8bitAO, AdamW4bitAO, AdamWfp8AO

# NEW — defers until optimizer instantiation
class _LazyAO:
    _resolved: dict | None = None
    def __init__(self, name): self._name = name
    @property
    def __name__(self) -> str: return self._name
    @property
    def __module__(self) -> str: return "LoraEasyCustomOptimizer.adam"
    @property
    def __qualname__(self) -> str: return self._name
    def __call__(self, *args, **kwargs): return self._resolve()(*args, **kwargs)
    def _resolve(self):
        if _LazyAO._resolved is None:
            from LoraEasyCustomOptimizer.adam import AdamW8bitAO, AdamW4bitAO, AdamWfp8AO
            _LazyAO._resolved = {...}
        return _LazyAO._resolved[self._name]

AdamW8bitAO = _LazyAO("AdamW8bitAO")
AdamW4bitAO = _LazyAO("AdamW4bitAO")
AdamWfp8AO = _LazyAO("AdamWfp8AO")
```

The proxy exposes `__name__`, `__module__`, `__qualname__` so the
`OPTIMIZERS` dict construction and `validation.py`'s module-path formatting
work without triggering the real import. The real class only resolves when
`optimizer_class(params, lr=...)` is called at training time.

### C. Standard-optimizer fast-path in `validation.py`

Added a frozenset of standard bare-name optimizers and a `dadapt*` prefix
check. If the optimizer matches, we `return` before the `from
LoraEasyCustomOptimizer import OPTIMIZERS` import runs, saving even the
80-module roster load for the common case.

## Import Chain After Fix

```
Training request with AdamW:
  validation.py → _STANDARD_OPTIMIZERS check → return early
  → LoraEasyCustomOptimizer never imported → torchao never touched ✓

Training request with CAME:
  validation.py → import LoraEasyCustomOptimizer
  → __init__.py imports 80+ modules (as before)
  → BUT adam.py is NEVER imported at load time
  → torchao never touched ✓

Training request with AdamW8bitAO:
  validation.py → import LoraEasyCustomOptimizer → OPTIMIZERS dict built
  → _LazyAO proxy in dict, __module__/__qualname__ returned as strings
  → training: importlib("LoraEasyCustomOptimizer.adam") → real class
  → optimizer = AdamW8bitAO(params, ...) → torchao FIRST import here ✓

Merge script (block_weight_merge.py):
  → imports torch + safetensors only
  → never touches LoraEasyCustomOptimizer → torchao never touched ✓
```

## Files Changed

| File | Change |
|------|--------|
| `custom_scheduler/LoraEasyCustomOptimizer/__init__.py` | Replaced eager AO import with `_LazyAO` proxy; changed `AdamW8bitKahan` import to `adam_kahan` |
| `custom_scheduler/LoraEasyCustomOptimizer/adam_kahan.py` | **New** — `AdamW8bitKahan` extracted from `adam.py` (bitsandbytes only, no torchao) |
| `utils/validation.py` | Standard-optimizer fast-path before OPTIMIZERS import |
