# Upstream Bug Fixes — 2026-05-05

Bugs found by CodeRabbit on PR #361 and fixed in our vendored code.
All of these originate in upstream repositories and should be contributed back.

---

## sd_scripts fixes → contribute to https://github.com/67372a/sd-scripts

### 1. `library/anima_vae.py` — API inconsistency + frame-drop bug

**Files:** `library/anima_vae.py`

**Bug 1 — `encode()` returns only `mu` but callers unpack `(mu, log_var)`**
`forward()` and `sample()` called `self.encode(x)` expecting a tuple but `encode()` only returned `mu`. Crash on first call.

Fix: `encode()` now returns `(mu, log_var)` and has `scale=(0.0, 1.0)` default.
`forward()`, `decode()`, and `sample()` updated with matching `scale` defaults.

**Bug 2 — Floor division drops frames on encode**
`iter_ = 1 + (t - 1) // 4` floors the chunk count, silently dropping the last
1–3 frames for any input with 6–8, 10–12, etc. frames.

Fix: `iter_ = 1 + (max(t - 1, 0) + 3) // 4` (ceiling division).

---

### 2. `networks/lora_diffusers.py` — RamTorch `.cpu()` on list wrapper

**File:** `networks/lora_diffusers.py` line 186

`self.org_module` is stored as `[org_module]` (a list) to prevent it becoming a
registered submodule. `self.org_module.cpu()` raises `AttributeError` — lists
have no `.cpu()`. Every other access in the file correctly uses `self.org_module[0]`.

Fix: `self.org_module[0].cpu()`

---

### 3. `networks/oft_flux.py` — same RamTorch list bug

**File:** `networks/oft_flux.py` line 97

Same pattern: `self.org_module.cpu()` → `self.org_module[0].cpu()`

---

### 4. `networks/oft.py` — same RamTorch list bug

**File:** `networks/oft.py` line 88

Same pattern: `self.org_module.cpu()` → `self.org_module[0].cpu()`

---

### 5. `finetune/tag_images_by_wd14_tagger.py` — variable shadowing destroys tag list

**File:** `finetune/tag_images_by_wd14_tagger.py` lines 249–259, 282–284

**Bug 1 — Variable shadowing:** Loop variable `for tag_replacements_arg in tag_replacements`
reused the outer parameter name. `tags = tag_replacements_arg.split(",")` then
overwrote the input `tags` list with the split of the replacement spec — every iteration
after the first processed garbage instead of actual tags.

Fix: Renamed loop variable to `replacement_spec`, split result to `parts`.

**Bug 2 — Return value discarded:** `process_tag_replacement(rating_tags, ...)` was
called without assigning the return value. The function returns the modified list
but does not modify in place, so all replacements were silently lost.

Fix: `rating_tags = process_tag_replacement(rating_tags, ...)` (and same for
`general_tags`, `character_tags`).

---

### 6. `sdxl_train.py` — `masks_reshaped[1]` IndexError when no attention masks

**File:** `sdxl_train.py` lines 984 and 1127

`masks_reshaped` is initialized to `[]` and only populated conditionally. Passing
`encoder_attention_mask=masks_reshaped[1]` crashes with `IndexError` when the list
is empty (normal non-masked training).

Fix: `encoder_attention_mask=masks_reshaped[1] if len(masks_reshaped) > 1 else None`
(applied at both call sites).

---

## LyCORIS fixes → contribute to https://github.com/67372a/LyCORIS

### 7. `lycoris/modules/full.py` — `None.dtype` AttributeError

**File:** `lycoris/modules/full.py` line 235

`diff_b` is initialized to `None` on line 230. The dtype guard at line 235
compared `org_bias.dtype != diff_b.dtype` before `diff_b` was assigned a tensor
value — guaranteed `AttributeError` on every call to `get_diff_weight` when
`self.bias is not None`.

Fix: Compare against `self.bias.dtype` (the correct reference) instead of
the not-yet-assigned `diff_b.dtype`.

---

### 8. `test/regex_args.py` — wrong preset key names in test fixture

**File:** `test/regex_args.py` line 40–41

`reset_globals()` passed `"reg_dims"` and `"reg_lrs"` to `apply_preset()`, but
the validated key names are `"network_reg_dims"` and `"network_reg_lrs"`. Every
test that calls this fixture in cleanup raises `KeyError`, breaking the entire
test file.

Fix: `"reg_dims"` → `"network_reg_dims"`, `"reg_lrs"` → `"network_reg_lrs"`

---

---

## Batch 2 fixes (sd_scripts → contribute to https://github.com/67372a/sd-scripts)

### 9. `sd3_train_network.py`, `lumina_train_network.py`, `hunyuan_image_train_network.py` — 4-tuple vs 5-tuple mismatch

**Files:** `sd3_train_network.py` line ~610, `lumina_train_network.py` line ~420, `hunyuan_image_train_network.py` line ~580

`train_network.py`'s base class was updated to unpack 5 values from `get_noise_pred_and_target`
(adding `noise` as the 5th element) but SD3, Lumina, and HunyuanImage overrides still
returned only 4 values. Every training run using these models crashes with a `ValueError`
on unpacking.

Fix: Added `noise` to all three return statements:
`return model_pred, target, timesteps, weighting` → `return model_pred, target, timesteps, weighting, noise`

---

### 10. `library/edm2_loss.py` — trailing comma stores tuple instead of bool

**File:** `library/edm2_loss.py`

`self.use_importance_weights=use_importance_weights,` — the trailing comma wraps the
value in a tuple `(True,)`. Any downstream truthiness check like `if self.use_importance_weights:`
still passes, but identity/equality checks (e.g. `== True`) fail silently.

Fix: Remove the trailing comma.

---

### 11. `library/strategy_sdxl.py` — `UnboundLocalError` on `attn1`/`attn2`

**File:** `library/strategy_sdxl.py`

`attn1` and `attn2` were referenced after an `if self.is_weighted:` block without
being initialised before it. When `is_weighted` is False the variables are never
bound, causing `UnboundLocalError` on the first non-weighted call.

Fix: Added `attn1 = None` and `attn2 = None` before the block.

---

### 12. `tools/cache_latents_standalone.py` — invalid autocast dtype for CUDA

**File:** `tools/cache_latents_standalone.py`

`fast_autocast_dtype = torch.float32` — CUDA's autocast context manager only accepts
`torch.float16` or `torch.bfloat16`; passing `torch.float32` raises a `RuntimeError`
whenever the latent cache tool runs with CUDA.

Fix: `fast_autocast_dtype = None` (disables the context without crashing).

---

### 13. `networks/oft.py` — RamTorch bias not transferred to device

**File:** `networks/oft.py`

In the RamTorch path, `org_module.bias` was passed directly to `F.conv2d`/`F.linear`
while the weight had already been transferred via `transfer_ramtensor_to_device`.
The bias remains on CPU, causing a device mismatch crash on the first forward pass
when bias is non-None.

Fix: Added `B = transfer_ramtensor_to_device(org_module.bias, x.device).to(dtype=org_dtype) if org_module.bias is not None else None`
and passed `B` to the functional calls.

---

### 14. `networks/oft_flux.py` — same bias device mismatch + None guard

**File:** `networks/oft_flux.py`

Same as item 13 for the flux variant. Additionally, `B1 = B[d1:d2].to(org_dtype)` was
called without guarding `B is not None`, causing `AttributeError` on modules without bias.

Fix: `B = transfer_ramtensor_to_device(org_module.bias, ...) if org_module.bias is not None else None`,
`B1 = B[d1:d2].to(org_dtype) if B is not None else None`.

---

### 15. `sdxl_train.py` — `torch.cuda.get/set_rng_state` called on MPS device

**File:** `sdxl_train.py` `switch_rng_state`/`restore_rng_state` functions

The MPS branch in both `switch_rng_state` and `restore_rng_state` called
`torch.cuda.get_rng_state()` / `torch.cuda.set_rng_state()` — these are CUDA-only
APIs and raise `AssertionError` on Apple Silicon. MPS training validation would always
crash at the RNG save/restore step.

Fix: `torch.cuda.get_rng_state()` → `torch.mps.get_rng_state()`,
`torch.cuda.set_rng_state(gpu_rng_state)` → `torch.mps.set_rng_state(gpu_rng_state)`.

---

### 16. `sdxl_train.py` — `del accelerator` before EDM2 save block

**File:** `sdxl_train.py` end of `train()`

The EDM2 importance-weights save block referenced `accelerator` after `del accelerator`
had already executed, causing `NameError` on any SDXL run with EDM2 adaptive loss
when saving at end of training.

Fix: Moved the EDM2 save block to before the `del accelerator` line.

---

### 17. `hunyuan_image_minimal_inference.py` — wrong arguments to `merge_lora_weights`

**File:** `hunyuan_image_minimal_inference.py` line 312

`merge_lora_weights(lora_hunyuan_image, model, args, device)` passed the entire
`argparse.Namespace` as `lora_weights` and `device` as `lora_multipliers`.
The function expects individual fields: `lora_weights: List[str]`,
`lora_multipliers: List[float]`, `include_patterns`, `exclude_patterns`, `device`, etc.
Every LyCORIS weight merge on HunyuanImage inference would crash with a `TypeError`.

Fix: Expanded to the correct keyword call:
```python
merge_lora_weights(
    lora_hunyuan_image, model,
    args.lora_weight, args.lora_multiplier,
    args.include_patterns, args.exclude_patterns,
    device, lycoris=True, save_merged_model=args.save_merged_model,
)
```

---

### 18. `tools/cache_text_encoder_outputs.py` — orphan expression statement

**File:** `tools/cache_text_encoder_outputs.py` line 209

`image_info` appeared as a bare expression statement with no effect — likely a debug
leftover. It is a no-op but triggers linters and misleads readers into thinking
something meaningful happens there.

Fix: Removed the orphan line.

---

### 19. `networks/lora_diffusers.py` — `logger.error` for optional import

**File:** `networks/lora_diffusers.py` line 25

`logger.error("Failed to import ramtorch …")` in a bare `except ImportError` block
fires at `ERROR` level whenever ramtorch is not installed — which is the normal state
for most users. This pollutes logs with a scary error for an optional dependency.

Fix: Changed to `logger.debug("ramtorch not available; CPU-bouncing linear disabled.")`.

---

## LyCORIS batch 2 fixes → contribute to https://github.com/67372a/LyCORIS

### 20. `test/regex_args.py` — same `net` instance passed to two `create_lycoris` calls

**File:** `test/regex_args.py` `test_exclude_patterns_empty_list_excludes_nothing`

Both `lycoris_all` and `lycoris_empty_exclude` were created from the same `net`
instance. `create_lycoris` wraps the model's modules in place; the second call wraps
already-wrapped modules, producing inflated lora counts and a meaningless comparison.

Fix: Created a separate `net_all = SimpleNet()` and `net_empty = SimpleNet()` for each call.

---

## Notes

- All fixes are one-liners or minimal surgical changes — no behaviour changes beyond the bug fix.
- The RamTorch `[0]` bug (batch 1, items 2–4) is systemic — check all other network files
  that store `org_module` as a list for the same pattern if adding new network types.
- Items 5 and 6 are independent of RamTorch and affect all users of tag replacement
  and SDXL validation respectively.
- The 5-tuple mismatch (item 9) will affect any new model type added to the codebase
  until the base class contract is documented — add `noise` to return values.
