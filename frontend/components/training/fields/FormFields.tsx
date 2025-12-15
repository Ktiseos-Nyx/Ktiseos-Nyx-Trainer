/**
 * Reusable Form Field Components
 * Built on shadcn/ui Form primitives + React Hook Form
 *
 * These components provide:
 * - Automatic validation display
 * - Accessible labels and error messages
 * - Consistent styling
 * - Help text support
 * - Type safety
 */

'use client';

import { UseFormReturn, FieldValues, Path } from 'react-hook-form';
import {
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Checkbox } from '@/components/ui/checkbox';
import { Textarea } from '@/components/ui/textarea';
import { Slider } from '@/components/ui/slider';

/**
 * Base props for all form fields
 */
interface BaseFieldProps<T extends FieldValues> {
  form: UseFormReturn<T>;
  name: Path<T>;
  label: string;
  description?: string;
  placeholder?: string;
  disabled?: boolean;
  readOnly?: boolean; // <--- ADD THIS LINE
}

/**
 * Text Input Field
 *
 * Usage:
 * ```tsx
 * <TextFormField
 *   form={form}
 *   name="project_name"
 *   label="Project Name"
 *   description="Alphanumeric characters, underscores, and hyphens only"
 *   placeholder="my_awesome_lora"
 * />
 * ```
 */
export function TextFormField<T extends FieldValues>({
  form,
  name,
  label,
  description,
  placeholder,
  disabled,
  readOnly, // <--- ACCEPT THE NEW PROP
}: BaseFieldProps<T>) {
  return (
    <FormField
      control={form.control}
      name={name}
      render={({ field }) => (
        <FormItem>
          <FormLabel>{label}</FormLabel>
          <FormControl>
            <Input
              placeholder={placeholder}
              disabled={disabled}
              readOnly={readOnly} // <--- PASS IT TO THE INPUT
              {...field}
              value={field.value ?? ''}
            />
          </FormControl>
          {description && <FormDescription>{description}</FormDescription>}
          <FormMessage />
        </FormItem>
      )}
    />
  );
}

/**
 * Number Input Field
 *
 * Usage:
 * ```tsx
 * <NumberFormField
 *   form={form}
 *   name="network_dim"
 *   label="Network Dimension"
 *   description="Higher = more detail but larger file size"
 *   min={1}
 *   max={1024}
 * />
 * ```
 */
export function NumberFormField<T extends FieldValues>({
  form,
  name,
  label,
  description,
  placeholder,
  disabled,
  min,
  max,
  step,
}: BaseFieldProps<T> & {
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <FormField
      control={form.control}
      name={name}
      render={({ field }) => (
        <FormItem>
          <FormLabel>{label}</FormLabel>
          <FormControl>
            <Input
              type="number"
              placeholder={placeholder}
              disabled={disabled}
              min={min}
              max={max}
              step={step}
              {...field}
              value={field.value ?? ''}
              onChange={(e) => {
                const value = e.target.value;
                field.onChange(value === '' ? undefined : Number(value));
              }}
            />
          </FormControl>
          {description && <FormDescription>{description}</FormDescription>}
          <FormMessage />
        </FormItem>
      )}
    />
  );
}

/**
 * Select Dropdown Field
 *
 * Usage:
 * ```tsx
 * <SelectFormField
 *   form={form}
 *   name="model_type"
 *   label="Model Type"
 *   description="Choose the model architecture"
 *   options={[
 *     { value: 'SDXL', label: 'SDXL' },
 *     { value: 'SD1.5', label: 'SD 1.5' },
 *     { value: 'Flux', label: 'Flux' },
 *   ]}
 * />
 * ```
 */
export function SelectFormField<T extends FieldValues>({
  form,
  name,
  label,
  description,
  placeholder = 'Select an option...',
  disabled,
  options,
}: BaseFieldProps<T> & {
  options: Array<{ value: string; label: string; description?: string }>;
}) {
  return (
    <FormField
      control={form.control}
      name={name}
      render={({ field }) => (
        <FormItem>
          <FormLabel>{label}</FormLabel>
          <Select
            onValueChange={field.onChange}
            defaultValue={field.value}
            disabled={disabled}
          >
            <FormControl>
              <SelectTrigger>
                <SelectValue placeholder={placeholder} />
              </SelectTrigger>
            </FormControl>
            <SelectContent>
              {options.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  <div className="flex flex-col">
                    <span>{option.label}</span>
                    {option.description && (
                      <span className="text-xs text-muted-foreground">
                        {option.description}
                      </span>
                    )}
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {description && <FormDescription>{description}</FormDescription>}
          <FormMessage />
        </FormItem>
      )}
    />
  );
}

/**
 * Checkbox Field
 *
 * Usage:
 * ```tsx
 * <CheckboxFormField
 *   form={form}
 *   name="flip_aug"
 *   label="Horizontal Flip Augmentation"
 *   description="Randomly flip images horizontally during training"
 * />
 * ```
 */
export function CheckboxFormField<T extends FieldValues>({
  form,
  name,
  label,
  description,
  disabled,
}: BaseFieldProps<T>) {
  return (
    <FormField
      control={form.control}
      name={name}
      render={({ field }) => (
        <FormItem className="flex flex-row items-start space-x-3 space-y-0 rounded-md border p-4">
          <FormControl>
            <Checkbox
              checked={field.value}
              onCheckedChange={field.onChange}
              disabled={disabled}
            />
          </FormControl>
          <div className="space-y-1 leading-none">
            <FormLabel>{label}</FormLabel>
            {description && <FormDescription>{description}</FormDescription>}
          </div>
          <FormMessage />
        </FormItem>
      )}
    />
  );
}

/**
 * Textarea Field (for long text)
 *
 * Usage:
 * ```tsx
 * <TextareaFormField
 *   form={form}
 *   name="sample_prompts"
 *   label="Sample Prompts"
 *   description="One prompt per line"
 *   placeholder="1girl, solo, smiling"
 *   rows={5}
 * />
 * ```
 */
export function TextareaFormField<T extends FieldValues>({
  form,
  name,
  label,
  description,
  placeholder,
  disabled,
  rows = 3,
}: BaseFieldProps<T> & {
  rows?: number;
}) {
  return (
    <FormField
      control={form.control}
      name={name}
      render={({ field }) => (
        <FormItem>
          <FormLabel>{label}</FormLabel>
          <FormControl>
            <Textarea
              placeholder={placeholder}
              disabled={disabled}
              rows={rows}
              {...field}
              value={field.value ?? ''}
            />
          </FormControl>
          {description && <FormDescription>{description}</FormDescription>}
          <FormMessage />
        </FormItem>
      )}
    />
  );
}

/**
 * Slider Field (for visual range selection)
 *
 * Usage:
 * ```tsx
 * <SliderFormField
 *   form={form}
 *   name="caption_dropout_rate"
 *   label="Caption Dropout Rate"
 *   description="Probability of dropping captions (0-1)"
 *   min={0}
 *   max={1}
 *   step={0.01}
 * />
 * ```
 */
export function SliderFormField<T extends FieldValues>({
  form,
  name,
  label,
  description,
  disabled,
  min = 0,
  max = 100,
  step = 1,
}: BaseFieldProps<T> & {
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <FormField
      control={form.control}
      name={name}
      render={({ field }) => (
        <FormItem>
          <div className="flex items-center justify-between">
            <FormLabel>{label}</FormLabel>
            <span className="text-sm text-muted-foreground">
              {field.value ?? min}
            </span>
          </div>
          <FormControl>
            <Slider
              min={min}
              max={max}
              step={step}
              value={[field.value ?? min]}
              onValueChange={(value) => field.onChange(value[0])}
              disabled={disabled}
            />
          </FormControl>
          {description && <FormDescription>{description}</FormDescription>}
          <FormMessage />
        </FormItem>
      )}
    />
  );
}
// --- PASTE THIS AT THE END OF FormFields.tsx ---

import { Combobox } from '@/components/ui/combobox'; // Make sure this path is correct

/**
 * Combobox Field (for searchable dropdowns)
 *
 * Usage:
 * ```tsx
 * <ComboboxFormField
 *   form={form}
 *   name="pretrained_model_name_or_path"
 *   label="Base Model"
 *   options={[{ value: '/path/model.safetensors', label: 'model.safetensors' }]}
 * />
 * ```
 */
export function ComboboxFormField<T extends FieldValues>({
  form,
  name,
  label,
  description,
  placeholder,
  disabled,
  options,
}: BaseFieldProps<T> & {
  options: Array<{ value: string; label: string }>;
}) {
  return (
    <FormField
      control={form.control}
      name={name}
      render={({ field }) => (
        <FormItem>
          <FormLabel>{label}</FormLabel>
          <FormControl>
            <Combobox
              options={options}
              value={field.value}
              onSelect={(currentValue) => {
                // Allows selecting and de-selecting
                form.setValue(name, currentValue === field.value ? "" : currentValue, { shouldValidate: true });
              }}
              placeholder={placeholder || 'Select a file...'}
              disabled={disabled}
            />
          </FormControl>
          {description && <FormDescription>{description}</FormDescription>}
          <FormMessage />
        </FormItem>
      )}
    />
  );
}
