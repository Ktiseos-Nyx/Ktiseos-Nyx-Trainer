# Form Primitives Guide

## 🎯 Why Use shadcn/ui Form Primitives?

### The Old Way (Manual Everything)
```tsx
// 25 lines of repetitive code per field
<div className="space-y-2">
  <Label htmlFor="project_name" className={errors.project_name ? 'text-red-400' : ''}>
    Project Name
    {errors.project_name && <span className="ml-2 text-xs">⚠️</span>}
  </Label>
  <Input
    id="project_name"
    type="text"
    placeholder="my_awesome_lora"
    {...form.register('project_name')}
    className={errors.project_name ? 'border-red-500 focus-visible:ring-red-500' : ''}
  />
  {errors.project_name && (
    <p className="text-xs text-red-400 flex items-center gap-1">
      <AlertCircle className="h-3 w-3" />
      {errors.project_name.message}
    </p>
  )}
  {!errors.project_name && (
    <p className="text-xs text-gray-500">
      Alphanumeric characters, underscores, and hyphens only
    </p>
  )}
</div>

// Repeat for 132 fields... 😱
```

### The New Way (shadcn Form Primitives)
```tsx
// 7 lines - clean, reusable, type-safe
<TextFormField
  form={form}
  name="project_name"
  label="Project Name"
  description="Alphanumeric characters, underscores, and hyphens only"
  placeholder="my_awesome_lora"
/>

// Multiply by 132 fields = still clean! ✨
```

---

## 📦 Available Form Primitives

### 1. TextFormField
**For:** Project names, paths, general text
```tsx
<TextFormField
  form={form}
  name="project_name"
  label="Project Name"
  description="Alphanumeric characters, underscores, and hyphens only"
  placeholder="my_awesome_lora"
/>
```

### 2. NumberFormField
**For:** Dimensions, batch sizes, epochs, steps
```tsx
<NumberFormField
  form={form}
  name="network_dim"
  label="Network Dimension"
  description="Higher = more detail but larger file size"
  min={1}
  max={1024}
  step={1}
/>
```

### 3. SelectFormField
**For:** Model type, optimizer, scheduler, precision
```tsx
<SelectFormField
  form={form}
  name="model_type"
  label="Model Type"
  description="Choose the model architecture"
  options={[
    { value: 'SDXL', label: 'SDXL', description: 'Best for 1024x1024' },
    { value: 'SD1.5', label: 'SD 1.5', description: 'Classic 512x512' },
    { value: 'Flux', label: 'Flux', description: 'Experimental' },
  ]}
/>
```

### 4. CheckboxFormField
**For:** Boolean flags (flip_aug, cache_latents, etc.)
```tsx
<CheckboxFormField
  form={form}
  name="flip_aug"
  label="Horizontal Flip Augmentation"
  description="Randomly flip images horizontally during training"
/>
```

### 5. TextareaFormField
**For:** Sample prompts, long text
```tsx
<TextareaFormField
  form={form}
  name="sample_prompts"
  label="Sample Prompts"
  description="One prompt per line"
  placeholder="1girl, solo, smiling"
  rows={5}
/>
```

### 6. SliderFormField
**For:** Visual range selection (dropout rates, noise values)
```tsx
<SliderFormField
  form={form}
  name="caption_dropout_rate"
  label="Caption Dropout Rate"
  description="Probability of dropping captions (0-1)"
  min={0}
  max={1}
  step={0.01}
/>
```

---

## 🏗️ Building Configuration Cards

### Example: Project Setup Card

```tsx
// components/training/cards/ProjectSetupCard.tsx
'use client';

import { UseFormReturn } from 'react-hook-form';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { TextFormField, SelectFormField } from '../fields/FormFields';
import type { TrainingConfig } from '@/lib/api';

interface ProjectSetupCardProps {
  form: UseFormReturn<Partial<TrainingConfig>>;
}

export function ProjectSetupCard({ form }: ProjectSetupCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Project Setup</CardTitle>
        <CardDescription>
          Basic project information and model selection
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <TextFormField
          form={form}
          name="project_name"
          label="Project Name"
          description="Alphanumeric characters, underscores, and hyphens only"
          placeholder="my_awesome_lora"
        />

        <SelectFormField
          form={form}
          name="model_type"
          label="Model Type"
          description="Choose the base model architecture"
          options={[
            { value: 'SDXL', label: 'SDXL', description: '1024x1024, most popular' },
            { value: 'SD1.5', label: 'SD 1.5', description: '512x512, classic' },
            { value: 'Flux', label: 'Flux', description: 'Experimental, high quality' },
            { value: 'SD3', label: 'SD 3', description: 'Latest Stability AI' },
          ]}
        />

        <TextFormField
          form={form}
          name="pretrained_model_name_or_path"
          label="Base Model Path"
          description="Path to .safetensors or .ckpt file, or HuggingFace model ID"
          placeholder="/path/to/model.safetensors"
        />

        <TextFormField
          form={form}
          name="vae_path"
          label="VAE Path (Optional)"
          description="Custom VAE for better image quality"
          placeholder="/path/to/vae.safetensors"
        />
      </CardContent>
    </Card>
  );
}
```

**Result:** 50 lines for a complete, validated, accessible card!

---

## 🎨 Benefits

### 1. **Automatic Validation Display**
```tsx
// Error message appears automatically when validation fails
<TextFormField name="project_name" ... />
// User types "my lora!" → Shows: "Project name must contain only..."
```

### 2. **Accessibility Built-in**
- Labels linked to inputs (`htmlFor` automatic)
- Error messages announced to screen readers
- ARIA attributes set correctly
- Keyboard navigation works

### 3. **Consistent Styling**
- All fields look the same
- Errors shown the same way
- Help text formatted consistently
- No more "this field looks different" bugs

### 4. **Type Safety**
```tsx
// TypeScript ensures you can't use wrong field names
<TextFormField
  name="project_name" // ✅ Valid
  name="projet_name"  // ❌ TypeScript error!
/>
```

### 5. **DRY (Don't Repeat Yourself)**
```tsx
// Change error styling once, affects all 132 fields
// In FormMessage component:
className="text-[0.8rem] font-medium text-destructive"
```

---

## 📊 Code Reduction Stats

### Old Approach (Manual)
- **25 lines** per text field
- **30 lines** per select field
- **20 lines** per checkbox
- **132 fields** × 25 lines average = **3,300 lines of form code**

### New Approach (Primitives)
- **7 lines** per text field
- **10 lines** per select field
- **6 lines** per checkbox
- **132 fields** × 7 lines average = **924 lines of form code**

**Result: 72% less code!** 🎉

---

## 🔧 Advanced Usage

### Conditional Fields

```tsx
{form.watch('model_type') === 'Flux' && (
  <NumberFormField
    form={form}
    name="guidance_scale"
    label="Guidance Scale"
    description="Flux.1 dev guidance (typical: 3.5)"
    min={0}
    max={30}
    step={0.1}
  />
)}
```

### Field Groups (Grid Layout)

```tsx
<div className="grid grid-cols-2 gap-4">
  <NumberFormField
    form={form}
    name="network_dim"
    label="Network Dimension"
  />
  <NumberFormField
    form={form}
    name="network_alpha"
    label="Network Alpha"
  />
</div>
```

### Disabled Fields (Computed Values)

```tsx
<NumberFormField
  form={form}
  name="total_steps"
  label="Total Training Steps"
  description="Calculated from epochs × images × repeats / batch size"
  disabled={true} // Read-only computed value
/>
```

---

## 🎯 Migration Pattern

### Step 1: Identify Field Type

```tsx
// Old manual input
<Input
  type="text"
  value={config.project_name}
  onChange={...}
/>

// → Use TextFormField
```

### Step 2: Replace with Primitive

```tsx
// Before:
<div className="space-y-2">
  <Label>Project Name</Label>
  <Input {...register('project_name')} />
  {errors.project_name && <span>{errors.project_name.message}</span>}
</div>

// After:
<TextFormField
  form={form}
  name="project_name"
  label="Project Name"
/>
```

### Step 3: Add Description (Optional)

```tsx
<TextFormField
  form={form}
  name="project_name"
  label="Project Name"
  description="Alphanumeric characters, underscores, and hyphens only"
/>
```

---

## 🚀 Next Steps

1. ✅ shadcn Form primitives installed
2. ✅ Reusable field components created
3. ⏭️ Create configuration cards (one per logical grouping)
4. ⏭️ Create tab components (compose cards)
5. ⏭️ Build main TrainingConfig component (orchestrator)

**Estimated time to build all 132 fields:** 2-3 hours (vs 10-20 hours manual)

---

## 📚 Resources

- **shadcn/ui Form Docs:** https://ui.shadcn.com/docs/components/form
- **React Hook Form Docs:** https://react-hook-form.com/
- **Zod Validation:** https://zod.dev/

---

**Result: Clean, maintainable, accessible forms with 72% less code!** 🎉
