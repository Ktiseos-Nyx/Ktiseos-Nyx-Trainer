# TODO: Patterns & Features from Kohya SS GUI

**Purpose:** Reference material from bmaltais/kohya_ss GUI to identify useful patterns and missing features

**Status:** Research phase - items here are for future implementation, not immediate priorities

---

## 📸 Captioning Features

### BLIP/BLIP2 Captioning (Photography-Style Captions)
**Priority:** Medium | **Effort:** 2-3 hours

**What it is:**
- Alternative to WD14 tagging - generates natural language captions instead of tags
- BLIP uses image-to-text model for descriptive sentences
- Better for photography datasets vs anime/illustration (which WD14 excels at)

**Current Status:**
- Might already be in `custom/tag_images_by_wd14_tagger.py` script
- Need to verify if BLIP/BLIP2 models are available
- WD14 is more suitable for our current anime/illustration focus

**Implementation Notes:**
- Uses Google research model weights
- Supports beam search, top_p sampling, caption length constraints
- Would add as alternative captioning method in Dataset workflow

**References:**
- `kohya_gui/blip_caption_gui.py`
- `kohya_gui/blip2_caption_gui.py`

---

## ⚙️ Missing Training Parameters

### SDXL-Specific Parameters
**Priority:** Low-Medium | **Effort:** 30 min - 1 hour

Parameters they have that we might be missing:

```python
# Memory Optimization
fused_backward_pass: bool = False
  # Reduces GPU memory for finetune/dreambooth
  # Mutually exclusive with fused_optimizer_groups

fused_optimizer_groups: int = 0
  # Groups optimizer operations (4-10 recommended)
  # Saves memory, mutually exclusive with fused_backward_pass

# Compatibility
disable_mmap_load_safetensors: bool = False
  # Disables memory mapping for safetensors loading
  # Compatibility flag for certain systems
```

**Action:** Compare with our Training Config to see if we already have these

**Reference:** `kohya_gui/class_sdxl_parameters.py`

---

### Flux-Specific Parameters
**Priority:** Medium | **Effort:** 1-2 hours

**Missing Flux parameters:**

```python
# Flow Sampling
discrete_flow_shift: float = 3.0
  # For Euler Discrete Scheduler
  # Flux-specific flow-based sampling

timestep_sampling: str = "flux_shift"
  # Options: flux_shift, sigma, shift, sigmoid, uniform
  # Flux uses flow-based vs noise-based diffusion

# Text Encoder
t5xxl_max_token_length: int = 512
  # Range: 512-4096
  # T5-XXL specific token limit

# Architecture
split_qkv: bool = False
  # Split QKV projection layers option

# Block Training
double_blocks_to_train: str = ""
  # ~19 total double blocks
  # Granular block selection via indices/ranges

single_blocks_to_train: str = ""
  # ~38 total single blocks
  # Selective training of block types

# Memory (deprecated but still there)
blocks_to_swap: int = 0
  # CPU offload for memory management
```

**Why it matters:** Flux uses different architecture than SDXL (flow-based vs noise-based)

**Action:** Add to Training Config when we're ready to fully support Flux training

**Reference:** `kohya_gui/class_flux1.py`

---

## 🔧 UX/UI Patterns Worth Stealing

### Caption File Extension Selector
**Priority:** Low | **Effort:** 10 minutes

**What:** Dropdown to choose caption file extension
- Current: Hardcoded to `.txt`
- Kohya allows: `.txt`, `.cap`, `.caption`

**Why:** User preference/compatibility with other tools

**Where to add:** Dataset tagging interface

---

### Smart Directory Management
**Priority:** Low | **Effort:** Already have this!

**Pattern:** Refresh button updates folder listings without reload
- ✅ We already have this in our file browser
- Their implementation uses Gradio's update() pattern
- Our Next.js approach is already better

---

### Cascading Validation
**Priority:** Low | **Effort:** Pattern to adopt

**Pattern:** Sequential validation with specific error messages
```python
if not folder_exists:
    log("Error: Folder not found")
elif not extension_provided:
    log("Error: Please specify extension")
else:
    proceed()
```

**Current:** We do validation but could make error messages more specific

**Where:** Form validation across all pages

---

## 🛠️ Setup/Installation Improvements

### Hardware Detection & Validation
**Priority:** Medium | **Effort:** 3-4 hours

**What Kohya does:**
1. Python version enforcement (3.10.9 to <3.13.0)
2. Auto-detect GPU: CUDA/ROCm/Intel OneAPI
3. Report VRAM, architecture, compute capabilities
4. Graceful CPU fallback

**What we do:**
- Basic vendored backend verification
- No hardware detection
- No Python version checking

**Benefits:**
- Better error messages for incompatible systems
- Auto-configure based on hardware
- Warn about insufficient VRAM before training

**Reference:** `setup/setup_common.py`

---

### Better Logging System
**Priority:** Low-Medium | **Effort:** 2-3 hours

**What Kohya does:**
- Uses Rich library for formatted console output
- Timestamps on all logs
- Dual output (console + file)
- Detailed subprocess stdout/stderr capture

**What we do:**
- Basic Python logging
- Training logs via WebSocket
- Could improve setup/installation logging

**Benefits:**
- Easier debugging
- Better UX during installation
- Persistent log files for troubleshooting

**Reference:** `setup/setup_common.py` logging setup

---

### Dependency Version Verification
**Priority:** Low | **Effort:** 1-2 hours

**What Kohya does:**
- `installed()` function checks package presence AND version
- Supports `>=` and `==` version specifiers
- Fallback name matching (underscores to dashes)
- Conditional installation based on checks

**What we do:**
- Basic requirements.txt installation
- No version verification
- Hope for the best 🤞

**Benefits:**
- Catch version conflicts early
- Selective upgrades
- Better compatibility assurance

**Reference:** `setup/setup_common.py` dependency management

---

## 📝 Implementation Priority Recommendations

### Do Now (Already done! ✅)
- Caption management (prefix/postfix/find-replace) ✅
- Batch operations ✅
- Directory management ✅

### Next Week
- BLIP/BLIP2 captioning (verify if already in custom tagger script)
- Caption file extension selector
- Missing Flux parameters (if we're serious about Flux support)

### Next Month
- Hardware detection & validation
- Better logging system
- Dependency version verification

### Someday/Maybe
- SDXL fused backward pass parameters (niche optimization)
- Port their smart UI patterns to our Next.js interface

---

## 🔗 Reference Links

All files analyzed from: https://github.com/bmaltais/kohya_ss

**Caption GUIs:**
- `kohya_gui/blip_caption_gui.py`
- `kohya_gui/basic_caption_gui.py`
- `kohya_gui/blip2_caption_gui.py`

**Parameter Classes:**
- `kohya_gui/class_sdxl_parameters.py`
- `kohya_gui/class_flux1.py`

**Setup Scripts:**
- `setup/setup_common.py`
- `setup/setup_windows.py`
- `setup/setup_linux.py`
- `setup/docker_setup.py`

---

## 💡 Key Takeaways

1. **We're competitive!** Most caption features are already implemented
2. **Flux needs more love** - Missing several Flux-specific parameters
3. **Setup could be smarter** - Hardware detection would improve UX
4. **BLIP is different from WD14** - Photography captions vs anime tags
5. **Our Next.js UI is better than Gradio** - Keep our patterns, steal their features

---

**Last Updated:** 2025-11-30
**Next Review:** When ready to implement Flux training or BLIP captioning
