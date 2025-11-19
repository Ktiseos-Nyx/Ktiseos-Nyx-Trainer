# Training Config Migration Plan

## Current State Analysis

### What We Have (Next.js - 16 parameters)
- Basic model/dataset paths
- Resolution, batch size, steps/epochs
- Learning rate (single value)
- Network dim/alpha
- Gradient checkpointing
- Mixed precision, optimizer, scheduler
- LR warmup steps

### What's Missing (Jupyter - 100+ parameters)

#### 1. **Model Selection & Paths**
- ✅ model_path (pretrained_model_name_or_path)
- ✅ dataset_dir (train_data_dir)
- ❌ model_type (SD1.5, SDXL, Flux, SD3)
- ❌ vae_path
- ❌ clip_l_path (Flux/SD3 only)
- ❌ clip_g_path (Flux/SD3 only)
- ❌ t5xxl_path (Flux/SD3 only)
- ❌ continue_from_lora (resume training)
- ❌ wandb_key (experiment tracking)

#### 2. **Dataset & Training Settings**
- ✅ resolution
- ✅ train_batch_size
- ✅ max_train_steps (but missing max_train_epochs properly)
- ❌ num_repeats (critical for Kohya!)
- ❌ epochs (separate from max_train_steps)
- ❌ seed
- ❌ flip_aug (flip augmentation)
- ❌ shuffle_caption
- ❌ dataset_size (auto-detected)
- ❌ step_calculator (shows total steps calculation)

#### 3. **Learning Rate Configuration**
- ✅ learning_rate (but it's a single value!)
- ❌ unet_lr (separate UNet learning rate)
- ❌ text_encoder_lr (separate Text Encoder LR)
- ✅ lr_scheduler
- ❌ lr_scheduler_number (for restarts/polynomial)
- ❌ lr_warmup_ratio
- ✅ lr_warmup_steps
- ❌ lr_power (polynomial scheduler)

#### 4. **Advanced Training Techniques**
- ❌ min_snr_gamma_enabled
- ❌ min_snr_gamma (value: 5.0)
- ❌ ip_noise_gamma_enabled
- ❌ ip_noise_gamma
- ❌ multinoise
- ❌ multires_noise_discount
- ❌ noise_offset
- ❌ adaptive_noise_scale
- ❌ zero_terminal_snr (recommended for SDXL)

#### 5. **Caption & Token Management**
- ❌ keep_tokens
- ❌ clip_skip
- ❌ caption_dropout_rate
- ❌ caption_tag_dropout_rate
- ❌ caption_dropout_every_n_epochs
- ❌ keep_tokens_separator
- ❌ secondary_separator
- ❌ enable_wildcard

#### 6. **Bucketing & Resolution**
- ❌ enable_bucket
- ❌ sdxl_bucket_optimization
- ❌ min_bucket_reso
- ❌ max_bucket_reso
- ❌ bucket_no_upscale

#### 7. **Memory & Performance**
- ✅ gradient_checkpointing
- ❌ gradient_accumulation_steps
- ❌ max_grad_norm
- ❌ full_fp16
- ❌ vae_batch_size
- ❌ no_half_vae

#### 8. **Data Augmentation**
- ❌ random_crop
- ❌ flip_aug (already listed above)

#### 9. **LoRA Structure (HUGE section!)**
- ❌ lora_type (dropdown with 11 options!)
  - LoRA, LoCon, LoKR, DyLoRA, DoRA, LoHa, (IA)³, GLoRA, Native Fine-Tuning, Diag-OFT, BOFT
- ✅ network_dim
- ✅ network_alpha
- ❌ dim_from_weights (auto-determine)
- ❌ network_dropout
- ❌ conv_dim (for textures/details)
- ❌ conv_alpha
- ❌ factor (LoKR decomposition)
- ❌ train_norm (LyCORIS)
- ❌ block_dims (per-block configuration)
- ❌ block_alphas (per-block alpha)
- ❌ preset (module preset)

#### 10. **Optimizer Options**
- ✅ optimizer_type (but limited options)
- ❌ optimizer_args (JSON string for custom args)
- ❌ AdamW variants (AdamW, AdamW8bit, PagedAdamW8bit, PagedAdamW32bit, Lion8bit, PagedLion8bit, SGDNesterov8bit, DAdaptation, DAdaptAdam, DAdaptAdaGrad, DAdaptAdan, DAdaptSGD, DAdaptLion, Prodigy, Adafactor)

#### 11. **Caching & Optimization**
- ❌ cache_latents
- ❌ cache_latents_to_disk
- ❌ cache_text_encoder_outputs
- ❌ no_metadata (don't save metadata)

#### 12. **Saving & Checkpointing**
- ❌ save_every_n_epochs
- ❌ save_every_n_steps
- ❌ save_last_n_epochs
- ❌ save_last_n_epochs_state
- ❌ save_model_as (safetensors/ckpt/pt/diffusers)
- ✅ save_precision
- ❌ output_name (custom output name)

#### 13. **Sample Generation During Training**
- ❌ sample_every_n_epochs
- ❌ sample_every_n_steps
- ❌ sample_prompts (file path)
- ❌ sample_sampler (ddim, pndm, etc.)

#### 14. **Logging & Monitoring**
- ❌ logging_dir
- ❌ log_with (tensorboard, wandb)
- ❌ log_prefix

#### 15. **Config Validation & Warnings**
- ❌ config_warnings (auto-detect conflicts)
  - Text encoder caching vs shuffle caption
  - Text encoder caching vs text encoder training
  - Random crop vs latent caching
  - And more...

---

## Implementation Strategy

### Phase 1: Data Structure & Backend (Priority: Critical)

**Goal:** Ensure backend can handle all parameters

1. **Update API Models** (`api/routes/training.py`)
   - Expand `TrainingConfig` Pydantic model with ALL parameters
   - Add optional fields with proper defaults
   - Add validation logic for conflicts

2. **Update Frontend TypeScript Interface** (`frontend/lib/api.ts`)
   - Mirror all backend parameters
   - Add proper TypeScript types
   - Make most fields optional with sensible defaults

3. **Test Data Flow**
   - Ensure backend accepts full config
   - Verify training manager receives all params
   - Test config save/load

**Estimated Additions:** ~85 new parameters

---

### Phase 2: UI Organization (Priority: Critical)

**Goal:** Make 100+ parameters manageable with PROPER WEB UI

#### UI Structure - Keep It Simple!

**We'll use TABS for now** (can revisit layout during ShadCN phase)

```
┌──────────────────────────────────────────────────────────────────┐
│  Training Configuration                                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────┬─────────┬─────────┬──────┬──────────┬────────┬──────┐  │
│  │Setup│ Dataset │ LoRA    │ LR   │ Advanced │ Saving │ Logs │  │
│  └─────┴─────────┴─────────┴──────┴──────────┴────────┴──────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Tab Content (2-column grid layout)                         │ │
│  │                                                             │ │
│  │  Left Column              Right Column                     │ │
│  │  ┌─────────────┐          ┌─────────────┐                 │ │
│  │  │   Fields    │          │   Fields    │                 │ │
│  │  └─────────────┘          └─────────────┘                 │ │
│  │                                                             │ │
│  │  (Smart 2-column responsive grid - not cramped!)           │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ⚠️ Config Warnings Box (appears when conflicts detected)       │
│                                                                  │
│  [Start Training]  [Save Config]  [Load Config]                 │
└──────────────────────────────────────────────────────────────────┘
```

#### Tab Organization (7 Clean Tabs)

**Tab 1: Setup** (Project & Model)
- Grid: 2 columns
- Project name, Model type dropdown
- Model path, VAE path
- Conditional: Flux/SD3 paths (clip_l, clip_g, t5xxl) - only show when model_type is Flux/SD3
- Continue from LoRA, WandB key

**Tab 2: Dataset** (Dataset & Basic Training)
- Grid: 2 columns
- Dataset directory (with browse button)
- Auto-detected: Dataset size (read-only, updates on directory change)
- Resolution, Num Repeats, Epochs
- Batch size, Max train steps, Seed
- Data augmentation: Flip aug, Random crop, Shuffle captions
- **Step Calculator Widget** (big, prominent, auto-updates)
  ```
  ╔═══════════════════════════════════════╗
  ║  📊 Total Training Steps: 1,250       ║
  ║  50 images × 10 repeats × 5 epochs   ║
  ║  ÷ 2 batch size = 1,250 steps        ║
  ╚═══════════════════════════════════════╝
  ```

**Tab 3: LoRA** (LoRA Structure & Network)
- Grid: 2 columns
- LoRA Type (big dropdown with 11 options)
- Network dim/alpha
- Conv dim/alpha (for LoCon/LoHa)
- Network dropout
- Advanced: factor, dim_from_weights, train_norm
- Bucketing section:
  - Enable bucket, SDXL optimization
  - Min/max bucket resolution
  - No upscale option

**Tab 4: Learning Rate** (LR Configuration)
- Grid: 2 columns
- UNet LR, Text Encoder LR
- LR Scheduler (dropdown)
- Scheduler number, LR warmup ratio/steps
- LR power (polynomial)
- Optimizer section:
  - Optimizer type (huge dropdown)
  - Gradient accumulation steps
  - Max grad norm
  - Optimizer args (advanced text field)

**Tab 5: Advanced** (Advanced Training Techniques)
- Grid: 2 columns
- SNR/Noise section:
  - Min SNR gamma (checkbox + value)
  - IP noise gamma (checkbox + value)
  - Multi-noise, multi-res noise discount
  - Noise offset, adaptive noise scale
  - Zero terminal SNR
- Caption/Token section:
  - Keep tokens, Clip skip
  - Caption dropout rates
  - Caption dropout every N epochs
  - Separators, wildcard notation
- Memory/Performance section:
  - Gradient checkpointing, Mixed precision, Full FP16
  - VAE batch size, No half VAE
  - Caching: latents, text encoder

**Tab 6: Saving** (Checkpoints & Samples)
- Grid: 2 columns
- Output directory, Output name
- Save frequency: every N epochs/steps
- Keep last N epochs/states
- Save format, Save precision
- No metadata checkbox
- Sample generation:
  - Sample every N epochs/steps
  - Sample prompts file
  - Sample sampler

**Tab 7: Logging** (Logging & Monitoring)
- Grid: 2 columns (but simple, few fields)
- Logging directory
- Log with (tensorboard/wandb)
- Log prefix
- WandB key (if not in Setup tab)

#### Simple Tab Approach
- 7 tabs to organize ~100 parameters
- 2-column grid layout per tab (responsive)
- Clean, not cramped
- **UI design details** can be refined when we add ShadCN
- For now: just get all the parameters in there and working!

#### Basic Layout Pattern
```tsx
<TabContent>
  <div className="grid md:grid-cols-2 gap-6">
    {/* Left column */}
    <div className="space-y-4">
      <input />
      <select />
    </div>
    {/* Right column */}
    <div className="space-y-4">
      <input />
      <input />
    </div>
  </div>
</TabContent>
```

**Note:** Layout can be improved/changed during ShadCN phase. Right now we just need FUNCTIONAL.

---

### Phase 3: Step-by-Step Implementation

#### Step 3.1: Backend Parameter Addition
**File:** `api/routes/training.py`

```python
class TrainingConfig(BaseModel):
    # Project & Model
    project_name: str
    model_type: str = "SDXL"  # SD1.5, SDXL, Flux, SD3
    pretrained_model_name_or_path: str
    vae_path: Optional[str] = None
    clip_l_path: Optional[str] = None  # Flux/SD3
    clip_g_path: Optional[str] = None  # Flux/SD3
    t5xxl_path: Optional[str] = None   # Flux/SD3
    continue_from_lora: Optional[str] = None
    wandb_key: Optional[str] = None

    # Dataset & Basic Training
    train_data_dir: str
    resolution: int = 1024
    num_repeats: int = 10
    max_train_epochs: int = 10
    max_train_steps: int = 0
    train_batch_size: int = 4
    seed: int = 42

    # Learning Rates
    unet_lr: float = 5e-4
    text_encoder_lr: float = 1e-4
    lr_scheduler: str = "cosine"
    lr_scheduler_number: int = 3
    lr_warmup_ratio: float = 0.05
    lr_warmup_steps: int = 0
    lr_power: float = 1.0

    # LoRA Structure
    lora_type: str = "LoRA"
    network_dim: int = 16
    network_alpha: int = 8
    conv_dim: int = 16
    conv_alpha: int = 8
    network_dropout: float = 0.0
    dim_from_weights: bool = False
    factor: int = -1
    train_norm: bool = False

    # Optimizer
    optimizer_type: str = "AdamW8bit"
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    optimizer_args: Optional[str] = None

    # Data Augmentation
    flip_aug: bool = False
    random_crop: bool = False
    shuffle_caption: bool = True

    # Caption & Token
    keep_tokens: int = 0
    clip_skip: int = 2
    caption_dropout_rate: float = 0.0
    caption_tag_dropout_rate: float = 0.0
    caption_dropout_every_n_epochs: int = 0
    keep_tokens_separator: str = ""
    secondary_separator: str = ""
    enable_wildcard: bool = False

    # Bucketing
    enable_bucket: bool = True
    sdxl_bucket_optimization: bool = False
    min_bucket_reso: int = 256
    max_bucket_reso: int = 2048
    bucket_no_upscale: bool = False

    # Advanced Training
    min_snr_gamma_enabled: bool = True
    min_snr_gamma: float = 5.0
    ip_noise_gamma_enabled: bool = False
    ip_noise_gamma: float = 0.05
    multinoise: bool = False
    multires_noise_discount: float = 0.25
    noise_offset: float = 0.0
    adaptive_noise_scale: float = 0.0
    zero_terminal_snr: bool = False

    # Memory & Performance
    gradient_checkpointing: bool = True
    mixed_precision: str = "fp16"
    full_fp16: bool = False
    vae_batch_size: int = 1
    no_half_vae: bool = False
    cache_latents: bool = False
    cache_latents_to_disk: bool = False
    cache_text_encoder_outputs: bool = False

    # Saving & Checkpoints
    output_dir: str
    save_every_n_epochs: int = 1
    save_every_n_steps: int = 0
    save_last_n_epochs: int = 0
    save_last_n_epochs_state: int = 0
    save_model_as: str = "safetensors"
    save_precision: str = "fp16"
    output_name: str = ""
    no_metadata: bool = False

    # Sample Generation
    sample_every_n_epochs: int = 0
    sample_every_n_steps: int = 0
    sample_prompts: Optional[str] = None
    sample_sampler: str = "euler_a"

    # Logging
    logging_dir: Optional[str] = None
    log_with: Optional[str] = None
    log_prefix: Optional[str] = None
```

**Add Config Validation Function:**
```python
def validate_training_config(config: TrainingConfig) -> List[str]:
    """Check for config conflicts and return warnings"""
    warnings = []

    if config.cache_text_encoder_outputs and config.shuffle_caption:
        warnings.append("Cannot use Caption Shuffling with Text Encoder Caching")

    if config.cache_text_encoder_outputs and config.text_encoder_lr > 0:
        warnings.append("Cannot cache Text Encoder while training it (set Text LR to 0)")

    if config.random_crop and config.cache_latents:
        warnings.append("Cannot use Random Crop with Latent Caching")

    # Add more validation as needed

    return warnings
```

#### Step 3.2: Frontend TypeScript Interface
**File:** `frontend/lib/api.ts`

Mirror the backend exactly with TypeScript types.

#### Step 3.3: Build Accordion Components
**File:** `frontend/components/training/TrainingConfigAccordion.tsx`

Create reusable accordion sections for each category.

#### Step 3.4: Build the Form
**File:** `frontend/components/training/TrainingConfig.tsx`

- Use accordion components
- Wire up all state
- Add auto-calculation for steps
- Add config warnings display
- Add conditional rendering (Flux/SD3 paths only show when needed)

#### Step 3.5: Add Step Calculator Widget
Auto-update when dataset_dir, repeats, epochs, or batch_size changes:
```
📊 Total Steps: 1250
50 images × 10 repeats × 5 epochs ÷ 2 batch = 1250 steps
```

---

### Phase 4: ShadCN Integration (AFTER Phase 3)

**Why After?**
- Need to know exact UI structure first
- ShadCN components will replace raw HTML/Tailwind
- Will make accordions, inputs, selects, sliders much prettier

#### ShadCN Components Needed:
1. **Accordion** - For collapsible sections
2. **Input** - Text inputs
3. **Select** - Dropdowns
4. **Slider** - Range inputs
5. **Checkbox** - Boolean toggles
6. **Textarea** - Multi-line inputs
7. **Button** - Actions
8. **Alert** - Config warnings
9. **Card** - Section containers
10. **Tabs** - Alternative to accordions (if preferred)
11. **Tooltip** - Help text on hover
12. **Label** - Form labels

#### Installation Steps:
```bash
cd frontend
npx shadcn-ui@latest init
npx shadcn-ui@latest add accordion
npx shadcn-ui@latest add input
npx shadcn-ui@latest add select
npx shadcn-ui@latest add slider
npx shadcn-ui@latest add checkbox
npx shadcn-ui@latest add textarea
npx shadcn-ui@latest add button
npx shadcn-ui@latest add alert
npx shadcn-ui@latest add card
npx shadcn-ui@latest add tabs
npx shadcn-ui@latest add tooltip
npx shadcn-ui@latest add label
```

#### Refactor With ShadCN:
Replace all the manual form elements with ShadCN components for:
- Better accessibility
- Consistent styling
- Dark mode support
- Cleaner code

---

## Summary Timeline

### Immediate Next Steps:
1. ✅ Create this plan document
2. ⏳ Expand backend `TrainingConfig` model with all ~85 missing parameters
3. ⏳ Expand frontend TypeScript interface to match
4. ⏳ Build accordion-based UI with all parameters organized
5. ⏳ Add step calculator auto-update
6. ⏳ Add config validation warnings
7. ⏳ Test config save/load
8. ⏳ Test training start with full config

### After Full Config Working:
1. Install ShadCN
2. Replace all form elements with ShadCN components
3. Add tooltips/help text
4. Improve visual design
5. Add dark mode polish

---

## Notes & Considerations

### What We're NOT Adding (From Jupyter):
- **Huber Loss** - Deleted from Jupyter due to safetensors corruption
- **Preset configurations** - User can use templates instead
- **Some exotic LyCORIS parameters** - Can add later if needed

### Token Budget Warning:
This is a HUGE undertaking. If we run out of tokens:
- Prioritize getting the data structure right first
- UI can be ugly initially, just needs to be functional
- ShadCN can wait until next session

### Testing Strategy:
1. Test with minimal config (just required fields)
2. Test with full config (all fields populated)
3. Test config conflicts trigger warnings
4. Test Flux/SD3 conditional fields show/hide
5. Test step calculator updates correctly

---

## File Checklist

Files That Need Updates:
- [ ] `api/routes/training.py` - Expand TrainingConfig model
- [ ] `frontend/lib/api.ts` - Expand TypeScript interface
- [ ] `frontend/components/training/TrainingConfig.tsx` - Rebuild with accordions
- [ ] (Optional) `frontend/components/training/ConfigAccordion.tsx` - Reusable accordion component
- [ ] Test all changes work end-to-end

Files for Later (ShadCN):
- [ ] All form components replaced with ShadCN
- [ ] Theme configuration
- [ ] Dark mode testing

---

**Ready to start Phase 1?** Let me know and I'll begin expanding the backend models!
