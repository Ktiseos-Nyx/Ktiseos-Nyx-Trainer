# WandB + TensorBoard Logging UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface the existing logging config fields (log_with, logging_dir, log_prefix, wandb_key, log_tracker_name, wandb_run_name) in the training form and wire WANDB_API_KEY into the Kohya subprocess env.

**Architecture:** One new card component (LoggingCard) with conditional sections based on log_with selection, one new tab entry in TrainingConfig, and a single env var injection in kohya.py. All Zod schema, TOML generation, and Pydantic model fields already exist — this is purely UI + one backend line.

**Tech Stack:** React 19, react-hook-form, shadcn/ui, TypeScript, Python asyncio subprocess

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `services/trainers/kohya.py` | Modify | Inject WANDB_API_KEY into subprocess env |
| `frontend/components/training/cards/LoggingCard.tsx` | Create | New card with all logging fields |
| `frontend/components/training/tabs/index.tsx` | Modify | Export LoggingTab wrapping LoggingCard |
| `frontend/components/training/TrainingConfig.tsx` | Modify | Register "logging" tab trigger and content |

---

## Task 1: Wire WANDB_API_KEY into Kohya subprocess

**Files:**
- Modify: `services/trainers/kohya.py` (around line 120 — after PYTHONPATH is built)
- Test: `tests/test_api_plumbing.py` (or new `tests/test_kohya_env.py`)

- [ ] **Step 1: Write a failing test**

Add to `tests/test_api_plumbing.py` (or a new file if cleaner):

```python
def test_wandb_key_injected_into_env(monkeypatch):
    """WANDB_API_KEY must appear in subprocess env when wandb_key is set."""
    from services.trainers.kohya import KohyaTrainer
    from services.models.training import TrainingConfig

    config = TrainingConfig(
        project_name="test",
        model_type="SD1.5",
        model_path="/fake/model.safetensors",
        output_dir="/fake/output",
        wandb_key="wbtest-secret-key",
    )
    trainer = KohyaTrainer(config)
    env = trainer._build_env()
    assert env.get("WANDB_API_KEY") == "wbtest-secret-key"


def test_wandb_key_absent_when_not_set():
    """WANDB_API_KEY must not appear in env when wandb_key is empty."""
    from services.trainers.kohya import KohyaTrainer
    from services.models.training import TrainingConfig

    config = TrainingConfig(
        project_name="test",
        model_type="SD1.5",
        model_path="/fake/model.safetensors",
        output_dir="/fake/output",
    )
    trainer = KohyaTrainer(config)
    env = trainer._build_env()
    assert "WANDB_API_KEY" not in env
```

- [ ] **Step 2: Run test to verify it fails**

```
cd C:\Users\dusk\Development\Ktiseos-Nyx-Trainer\Ktiseos-Nyx-Trainer
python -m pytest tests/ -k "wandb_key" -v
```

Expected: `AttributeError: 'KohyaTrainer' object has no attribute '_build_env'` or similar — confirms the method doesn't exist yet.

- [ ] **Step 3: Refactor env setup into a `_build_env` method**

In `services/trainers/kohya.py`, extract the env block (currently inline in `run()`) into a new method and add the WANDB_API_KEY injection.

Find the block starting at line ~113:
```python
env = python_subprocess_env()
derrian_dir = str(self.sd_scripts_dir.parent)
custom_sched_dir = str(self.sd_scripts_dir.parent / "custom_scheduler")
existing_pythonpath = env.get("PYTHONPATH", "")
new_paths = f"{derrian_dir}{os.pathsep}{custom_sched_dir}"
env["PYTHONPATH"] = f"{new_paths}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else new_paths
```

Extract to a method and add WandB:

```python
def _build_env(self) -> dict:
    """Build subprocess environment with PYTHONPATH and optional credentials."""
    env = python_subprocess_env()
    derrian_dir = str(self.sd_scripts_dir.parent)
    custom_sched_dir = str(self.sd_scripts_dir.parent / "custom_scheduler")
    existing_pythonpath = env.get("PYTHONPATH", "")
    new_paths = f"{derrian_dir}{os.pathsep}{custom_sched_dir}"
    env["PYTHONPATH"] = f"{new_paths}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else new_paths
    if self.config.wandb_key:
        env["WANDB_API_KEY"] = self.config.wandb_key
    return env
```

Then in `run()`, replace the inline block with:
```python
env = self._build_env()
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/ -k "wandb_key" -v
```

Expected: `2 passed`

- [ ] **Step 5: Run the full test suite to check for regressions**

```
python -m pytest tests/ -v
```

Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add services/trainers/kohya.py tests/
git commit -m "feat: inject WANDB_API_KEY into Kohya subprocess env from config.wandb_key"
```

---

## Task 2: Create LoggingCard component

**Files:**
- Create: `frontend/components/training/cards/LoggingCard.tsx`

No automated tests for UI components in this project — verify manually after Task 4.

- [ ] **Step 1: Create `frontend/components/training/cards/LoggingCard.tsx`**

```tsx
/**
 * Logging Configuration Card
 * Configures TensorBoard and WandB logging for training runs.
 */

'use client';

import { UseFormReturn } from 'react-hook-form';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { SelectFormField, TextFormField } from '../fields/FormFields';
import type { TrainingConfig } from '@/lib/api';
import { BarChart2 } from 'lucide-react';

interface LoggingCardProps {
  form: UseFormReturn<TrainingConfig>;
}

export function LoggingCard({ form }: LoggingCardProps) {
  const logWith = form.watch('log_with');
  const showTensorboard = logWith === 'tensorboard' || logWith === 'all';
  const showWandB = logWith === 'wandb' || logWith === 'all';

  return (
    <Card className="border-blue-500/30">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BarChart2 className="h-5 w-5 text-blue-400" />
          Logging
        </CardTitle>
        <CardDescription>
          Send training metrics to TensorBoard or WandB
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <SelectFormField
          form={form}
          name="log_with"
          label="Log With"
          description="Where to send training metrics (loss, learning rate, etc.)"
          options={[
            { value: '', label: 'None' },
            { value: 'tensorboard', label: 'TensorBoard' },
            { value: 'wandb', label: 'WandB' },
            { value: 'all', label: 'Both' },
          ]}
        />

        {showTensorboard && (
          <div className="space-y-4 pt-2 border-t border-slate-700">
            <p className="text-sm font-semibold text-gray-300">TensorBoard</p>

            <TextFormField
              form={form}
              name="logging_dir"
              label="Log Directory"
              description="Directory where TensorBoard event files are written"
              placeholder="./logs"
            />

            <TextFormField
              form={form}
              name="log_prefix"
              label="Log Prefix (Optional)"
              description="Prefix for the timestamped log folder (e.g. 'sdxl-' → logs/sdxl-20260520123456)"
              placeholder="my-experiment-"
            />
          </div>
        )}

        {showWandB && (
          <div className="space-y-4 pt-2 border-t border-slate-700">
            <p className="text-sm font-semibold text-gray-300">WandB</p>

            <FormField
              control={form.control}
              name="wandb_key"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>API Key</FormLabel>
                  <FormControl>
                    <Input
                      type="password"
                      placeholder="wbapi-..."
                      {...field}
                      value={field.value ?? ''}
                    />
                  </FormControl>
                  <FormDescription>
                    Your WandB API key —{' '}
                    <a
                      href="https://wandb.ai/authorize"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="underline text-blue-400"
                    >
                      get it here
                    </a>
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <TextFormField
              form={form}
              name="log_tracker_name"
              label="Project Name (Optional)"
              description="WandB project that groups all your runs (e.g. 'sdxl-experiments')"
              placeholder="my-lora-project"
            />

            <TextFormField
              form={form}
              name="wandb_run_name"
              label="Run Name (Optional)"
              description="Name for this individual training run"
              placeholder="sdxl-v2-attempt-1"
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```
cd C:\Users\dusk\Development\Ktiseos-Nyx-Trainer\Ktiseos-Nyx-Trainer\frontend
npx tsc --noEmit --project tsconfig.json 2>&1 | grep "error TS"
```

Expected: same pre-existing errors as before (TrainingMonitor.tsx:180, config-service.ts:563), no new errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/components/training/cards/LoggingCard.tsx
git commit -m "feat: add LoggingCard component for WandB and TensorBoard config"
```

---

## Task 3: Register LoggingTab in tabs/index.tsx

**Files:**
- Modify: `frontend/components/training/tabs/index.tsx`

- [ ] **Step 1: Add LoggingCard import and LoggingTab export**

At the top of `frontend/components/training/tabs/index.tsx`, add the import alongside the others:

```tsx
import { LoggingCard } from '../cards/LoggingCard';
```

At the bottom of the file, after `SavingTab`, add:

```tsx
/**
 * Logging Tab
 * TensorBoard and WandB metric logging configuration
 */
export function LoggingTab({ form, onSave }: TabProps) {
  return (
    <div className="space-y-6">
      <LoggingCard form={form} />
    </div>
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```
cd C:\Users\dusk\Development\Ktiseos-Nyx-Trainer\Ktiseos-Nyx-Trainer\frontend
npx tsc --noEmit --project tsconfig.json 2>&1 | grep "error TS"
```

Expected: no new errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/components/training/tabs/index.tsx
git commit -m "feat: export LoggingTab from training tabs"
```

---

## Task 4: Wire LoggingTab into TrainingConfig

**Files:**
- Modify: `frontend/components/training/TrainingConfig.tsx`

- [ ] **Step 1: Add LoggingTab import**

Find the existing imports at the top of `frontend/components/training/TrainingConfig.tsx`:

```tsx
import {
  SetupTab,
  DatasetTab,
  LoRATab,
  LearningTab,
  PerformanceTab,
  AdvancedTab,
  SavingTab,
} from './tabs';
```

Add `LoggingTab` to the destructure:

```tsx
import {
  SetupTab,
  DatasetTab,
  LoRATab,
  LearningTab,
  PerformanceTab,
  AdvancedTab,
  SavingTab,
  LoggingTab,
} from './tabs';
```

- [ ] **Step 2: Add TabsTrigger**

Find the `<TabsList>` block (around line 269):

```tsx
<TabsTrigger value="setup">Setup</TabsTrigger>
<TabsTrigger value="dataset">Dataset</TabsTrigger>
<TabsTrigger value="lora">LoRA</TabsTrigger>
<TabsTrigger value="learning">Learning</TabsTrigger>
<TabsTrigger value="performance">Performance</TabsTrigger>
<TabsTrigger value="advanced">Advanced</TabsTrigger>
<TabsTrigger value="saving">Saving</TabsTrigger>
```

Add after "saving":

```tsx
<TabsTrigger value="logging">Logging</TabsTrigger>
```

- [ ] **Step 3: Add TabsContent**

Find the `<TabsContent value="saving">` line (around line 284):

```tsx
<TabsContent value="saving"><SavingTab form={form} onSave={handleCardSave} /></TabsContent>
```

Add immediately after:

```tsx
<TabsContent value="logging"><LoggingTab form={form} onSave={handleCardSave} /></TabsContent>
```

- [ ] **Step 4: Verify TypeScript compiles**

```
cd C:\Users\dusk\Development\Ktiseos-Nyx-Trainer\Ktiseos-Nyx-Trainer\frontend
npx tsc --noEmit --project tsconfig.json 2>&1 | grep "error TS"
```

Expected: no new errors.

- [ ] **Step 5: Manual smoke test**

Start the dev server:
```
cd C:\Users\dusk\Development\Ktiseos-Nyx-Trainer\Ktiseos-Nyx-Trainer\frontend
npm run dev
```

Navigate to `http://localhost:3000/training`. Verify:
- "Logging" tab appears after "Saving" in the tab strip
- Selecting "None" shows only the `log_with` dropdown
- Selecting "TensorBoard" reveals Log Directory and Log Prefix fields
- Selecting "WandB" reveals API Key (masked), Project Name, Run Name fields
- Selecting "Both" reveals all fields from both sections
- API Key field masks the input (type="password")
- The WandB "get it here" link opens `https://wandb.ai/authorize`

- [ ] **Step 6: Commit**

```bash
git add frontend/components/training/TrainingConfig.tsx
git commit -m "feat: register Logging tab in training form"
```

---

## Task 5: Push and open PR

- [ ] **Step 1: Push branch**

```bash
git push origin feat/wandb-tensorboard-logging-ui
```

- [ ] **Step 2: Open PR to main**

```bash
gh pr create \
  --repo Ktiseos-Nyx/Ktiseos-Nyx-Trainer \
  --base main \
  --head feat/wandb-tensorboard-logging-ui \
  --title "feat: add WandB + TensorBoard logging UI to training form" \
  --body "$(cat <<'EOF'
## Summary

- New **LoggingCard** in the training form with conditional sections for TensorBoard and WandB
- `WANDB_API_KEY` now injected into the Kohya subprocess env when `wandb_key` is set (LT-1)
- Closes issues LT-1 and UI-1 from BETA_PLANNING.md

## Fields added

| Field | Shown when |
|-------|-----------|
| log_with (None/TB/WandB/Both) | Always |
| logging_dir | TensorBoard or Both |
| log_prefix | TensorBoard or Both |
| wandb_key (password) | WandB or Both |
| log_tracker_name | WandB or Both |
| wandb_run_name | WandB or Both |

All fields already existed in the Zod schema, Pydantic model, and TOML generator — this PR adds only UI and the one missing env var.

## Test plan

- [ ] Python tests pass: `pytest tests/ -k "wandb_key" -v` — 2 passed
- [ ] TypeScript compiles with no new errors
- [ ] Logging tab appears in training form after Saving tab
- [ ] Conditional field visibility works correctly for all four log_with values
- [ ] WandB API key field is masked (type="password")

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
