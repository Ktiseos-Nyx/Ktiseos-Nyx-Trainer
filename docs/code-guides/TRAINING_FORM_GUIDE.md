# Training Form System Guide

> **The Ultimate UX-Friendly Training Configuration**
>
> Say goodbye to janky forms! This new system combines Zustand + React Hook Form + Zod for the best possible user experience.

## 🎯 What Makes This Better Than Other Trainers

### Problems with "Janky" LoRA Trainers:
- ❌ Lose all settings on page refresh
- ❌ No validation until you hit submit (and it fails)
- ❌ Have to reconfigure everything from scratch every time
- ❌ Cryptic error messages
- ❌ No way to save favorite configurations
- ❌ Forms with 100+ fields and no organization

### Our Solution:
- ✅ **Auto-save to localStorage** - Settings survive page refresh
- ✅ **Real-time validation** - See errors as you type
- ✅ **Preset system** - Save and load favorite configs
- ✅ **Helpful error messages** - Clear, actionable guidance
- ✅ **Built-in presets** - SDXL Character, SD1.5, Flux, etc.
- ✅ **Clean, organized UI** - Tabbed interface with logical grouping

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────┐
│           TrainingConfig Component              │
│  (React component with form UI)                 │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│          useTrainingForm() Hook                 │
│  (Combines all three libraries)                 │
└──┬──────────────┬──────────────┬────────────────┘
   │              │              │
   ▼              ▼              ▼
┌─────────┐  ┌──────────┐  ┌─────────┐
│ Zustand │  │ React    │  │   Zod   │
│ (Store) │  │ Hook     │  │ (Valid- │
│         │  │ Form     │  │  ation) │
└────┬────┘  └────┬─────┘  └────┬────┘
     │            │             │
     │            │             │
     ▼            ▼             ▼
localStorage   Form State   Schema Validation
```

### Components:

1. **Zustand Store** (`store/trainingStore.ts`)
   - Persists config to localStorage
   - Survives page refresh
   - Global state management

2. **React Hook Form** (`useTrainingForm` hook)
   - Form state management
   - Field registration
   - Submit handling

3. **Zod Validation** (`lib/validation.ts`)
   - TypeScript-first schemas
   - Real-time validation
   - Helpful error messages

4. **PresetManager** (`components/training/PresetManager.tsx`)
   - Save/load favorite configs
   - Built-in presets (SDXL, SD1.5, Flux)
   - Import/export presets

## 🚀 Quick Start

### Basic Usage

```tsx
import { useTrainingForm } from '@/hooks/useTrainingForm';

function MyTrainingForm() {
  // Initialize the form - that's it!
  const { form, config, isValid, onSubmit } = useTrainingForm();

  const startTraining = async (validatedConfig) => {
    // Config is already validated by Zod
    await trainingAPI.start(validatedConfig);
  };

  return (
    <form onSubmit={onSubmit(startTraining)}>
      {/* Register a field */}
      <Input {...form.register('project_name')} />

      {/* Show validation errors */}
      {form.formState.errors.project_name && (
        <span>{form.formState.errors.project_name.message}</span>
      )}

      <Button type="submit" disabled={!isValid}>
        Start Training
      </Button>
    </form>
  );
}
```

### With Auto-Save

```tsx
const { form, isDirty, saveConfig } = useTrainingForm({
  autoSave: true,           // Enable auto-save
  autoSaveDelay: 500,       // Save 500ms after user stops typing
  validateOnChange: true,   // Real-time validation
});

// Shows if user has unsaved changes
{isDirty && <Badge>Unsaved Changes</Badge>}

// Manual save button
<Button onClick={saveConfig}>Save</Button>
```

### With Presets

```tsx
import PresetManager from '@/components/training/PresetManager';

const { config, loadPreset } = useTrainingForm();

return (
  <PresetManager
    currentConfig={config}
    onLoadPreset={loadPreset}
  />
);
```

## 📝 Form Field Patterns

### Text Input with Validation

```tsx
<div className="space-y-2">
  <Label>Project Name</Label>
  <Input {...form.register('project_name')} />
  {form.formState.errors.project_name && (
    <p className="text-red-400 text-xs">
      {form.formState.errors.project_name.message}
    </p>
  )}
</div>
```

### Number Input

```tsx
<Input
  type="number"
  {...form.register('network_dim', { valueAsNumber: true })}
/>
```

### Select Dropdown

```tsx
<Select
  value={config.model_type}
  onValueChange={(value) => form.setValue('model_type', value)}
>
  <SelectTrigger>
    <SelectValue />
  </SelectTrigger>
  <SelectContent>
    <SelectItem value="SDXL">SDXL</SelectItem>
    <SelectItem value="SD1.5">SD 1.5</SelectItem>
  </SelectContent>
</Select>
```

### Checkbox

```tsx
<Checkbox
  checked={config.flip_aug}
  onCheckedChange={(checked) => form.setValue('flip_aug', checked)}
/>
```

## 🎨 Built-in Presets

The system includes pre-configured presets for common scenarios:

### SDXL Character
```typescript
{
  model_type: 'SDXL',
  resolution: 1024,
  network_dim: 32,
  network_alpha: 32,
  unet_lr: 0.0001,
  text_encoder_lr: 0.00005,
  optimizer_type: 'AdamW8bit',
  // ... optimized for character training
}
```

### SDXL Style
```typescript
{
  model_type: 'SDXL',
  resolution: 1024,
  network_dim: 64,
  network_alpha: 64,
  unet_lr: 0.0002,
  text_encoder_lr: 0.0001,
  optimizer_type: 'Prodigy',
  // ... optimized for art style training
}
```

### SD1.5 Character
```typescript
{
  model_type: 'SD1.5',
  resolution: 512,
  network_dim: 32,
  clip_skip: 1,
  // ... classic SD1.5 settings
}
```

### Flux (Experimental)
```typescript
{
  model_type: 'Flux',
  resolution: 1024,
  network_dim: 16,
  guidance_scale: 3.5,
  timestep_sampling: 'flux_shift',
  // ... experimental Flux settings
}
```

## 🔧 Validation Rules

### Project Name
- Must be 1-100 characters
- Alphanumeric + underscores + hyphens only
- Example: `my_awesome_lora`

### Resolution
- Range: 256 - 4096
- Must be divisible by 64
- Valid: 512, 768, 1024, 2048
- Invalid: 500, 1000

### Learning Rates
- Must be positive
- Maximum: 1.0
- Typical UNet LR: 0.0001 - 0.001
- Typical TE LR: 0.00001 - 0.0001

### Network Dimension
- Range: 1 - 1024
- Higher = more detail but larger file size
- Typical: 16, 32, 64, 128

### Batch Size
- Range: 1 - 128
- Lower = less VRAM
- Higher = faster training

## 📦 Custom Presets

### Save Current Config as Preset

```tsx
// User clicks "Save Current" in PresetManager
// Preset is saved to localStorage
// Available in all future sessions
```

### Export Presets (JSON)

```typescript
// Click "Export" to download presets as JSON
// Share with friends or backup
// Format:
[
  {
    "id": "custom_1234567890",
    "name": "My Character Settings",
    "description": "Perfect for anime characters",
    "config": { /* full config */ },
    "createdAt": 1234567890000
  }
]
```

### Import Presets

```typescript
// Click "Import" and select JSON file
// Presets are added to your library
// Duplicates are handled gracefully
```

## 🔄 Migration Guide (Existing Component)

If you have an existing training form using `useState`, here's how to migrate:

### Before (Old Way)
```tsx
const [config, setConfig] = useState({ /* defaults */ });

const handleChange = (field, value) => {
  setConfig(prev => ({ ...prev, [field]: value }));
};

const handleSubmit = async () => {
  // No validation!
  // Lost on page refresh!
  await trainingAPI.start(config);
};
```

### After (New Way)
```tsx
const { form, config, isValid, onSubmit } = useTrainingForm();

// That's it! You get:
// - Auto-save to localStorage
// - Real-time Zod validation
// - Form state management
// - Preset system

const startTraining = async (validatedConfig) => {
  await trainingAPI.start(validatedConfig);
};
```

### Step-by-Step Migration

1. **Replace useState with hook**
   ```diff
   - const [config, setConfig] = useState(defaults);
   + const { form, config } = useTrainingForm();
   ```

2. **Replace manual onChange handlers**
   ```diff
   - <Input onChange={e => setConfig({...config, name: e.target.value})} />
   + <Input {...form.register('project_name')} />
   ```

3. **Add validation display**
   ```tsx
   {form.formState.errors.project_name && (
     <span>{form.formState.errors.project_name.message}</span>
   )}
   ```

4. **Replace submit handler**
   ```diff
   - <form onSubmit={handleSubmit}>
   + <form onSubmit={onSubmit(startTraining)}>
   ```

## 📊 Example: Full Form Component

See `components/training/TrainingConfigExample.tsx` for a complete reference implementation showing:

- ✅ All form patterns (text, number, select, checkbox)
- ✅ Validation error display
- ✅ PresetManager integration
- ✅ Auto-save indicator
- ✅ Status badges (dirty, valid, error count)
- ✅ Organized tabs (Setup, Dataset, LoRA, Learning)

## 🐛 Troubleshooting

### "Config not saving to localStorage"

Make sure you're using the hook correctly:
```tsx
const { form } = useTrainingForm({ autoSave: true });
```

### "Validation errors not showing"

Check that you're displaying errors:
```tsx
{form.formState.errors.field_name?.message}
```

### "Form not resetting when loading preset"

The hook handles this automatically. Make sure you're using `loadPreset`:
```tsx
const { loadPreset } = useTrainingForm();
loadPreset(myPreset.config);
```

## 🎯 Best Practices

1. **Always show validation errors**
   - Users need to know what's wrong
   - Use red text and icons for visibility

2. **Provide helpful error messages**
   - Bad: "Invalid value"
   - Good: "Resolution must be divisible by 64 (e.g., 512, 768, 1024)"

3. **Use auto-save**
   - Prevent data loss from accidental refresh
   - 500ms delay is good balance

4. **Show save status**
   - Display "Unsaved Changes" badge
   - Show "Auto-saving..." indicator

5. **Organize fields logically**
   - Use tabs or sections
   - Group related settings

6. **Include built-in presets**
   - Helps new users get started
   - Showcases best practices

## 📚 API Reference

See code documentation in:
- `hooks/useTrainingForm.ts` - Main hook
- `store/trainingStore.ts` - Zustand store
- `lib/validation.ts` - Zod schemas
- `components/training/PresetManager.tsx` - Preset UI

## 🚀 Future Improvements

- [ ] Preset sharing via URL
- [ ] Validation warning levels (error vs warning)
- [ ] Undo/redo support
- [ ] Config diff viewer
- [ ] Training calculator integration
- [ ] Recommended settings based on VRAM

---

**Built with ❤️ to make LoRA training actually enjoyable!**
