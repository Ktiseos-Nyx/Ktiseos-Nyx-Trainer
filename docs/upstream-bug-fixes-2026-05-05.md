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

## Notes

- All fixes are one-liners or minimal surgical changes — no behaviour changes beyond the bug fix.
- The RamTorch `[0]` bug (items 2–4) is systemic — check all other network files
  that store `org_module` as a list for the same pattern if adding new network types.
- Items 5 and 6 are independent of RamTorch and affect all users of tag replacement
  and SDXL validation respectively.
