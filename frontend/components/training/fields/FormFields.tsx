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
import {
  Combobox,
  ComboboxAnchor,
  ComboboxInput,
  ComboboxTrigger,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxItem,
} from '@/components/ui/combobox';

interface BaseFieldProps<T extends FieldValues> {
  form: UseFormReturn<T>;
  name: Path<T>;
  label: string;
  description?: string;
  placeholder?: string;
  disabled?: boolean;
  readOnly?: boolean;
}

/**
 * Text Input Field
 */
export function TextFormField<T extends FieldValues>({
  form,
  name,
  label,
  description,
  placeholder,
  disabled,
  readOnly,
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
              readOnly={readOnly}
              {...field}
              value={field.value ?? ''}
              // ✅ FORCE SYNC ON CHANGE
              onChange={(e) => {
                field.onChange(e);
                form.setValue(name, e.target.value as any, { shouldDirty: true });
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
 * Number Input Field
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
                const val = e.target.value === '' ? undefined : Number(e.target.value);
                field.onChange(val);
                // ✅ FORCE SYNC ON CHANGE
                form.setValue(name, val as any, { shouldDirty: true });
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
            value={field.value}
            onValueChange={(val) => {
              field.onChange(val);
              // ✅ FORCE SYNC ON CHANGE
              form.setValue(name, val as any, { shouldDirty: true });
            }}
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
              onCheckedChange={(checked) => {
                field.onChange(checked);
                // ✅ FORCE SYNC ON CHANGE
                form.setValue(name, checked as any, { shouldDirty: true });
              }}
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
 * Textarea Field
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
              onChange={(e) => {
                field.onChange(e);
                // ✅ FORCE SYNC ON CHANGE
                form.setValue(name, e.target.value as any, { shouldDirty: true });
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
 * Slider Field
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
              onValueChange={(value) => {
                field.onChange(value[0]);
                // ✅ FORCE SYNC ON CHANGE
                form.setValue(name, value[0] as any, { shouldDirty: true });
              }}
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

/**
 * Combobox Field
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
          <Combobox
            value={field.value}
            onValueChange={(val) => {
              field.onChange(val);
              // ✅ FORCE SYNC ON CHANGE
              form.setValue(name, val as any, { shouldDirty: true });
            }}
            disabled={disabled}
          >
            <ComboboxAnchor>
              <FormControl>
                <ComboboxInput placeholder={placeholder || 'Select a file...'} />
              </FormControl>
              <ComboboxTrigger />
            </ComboboxAnchor>
            <ComboboxContent>
              <ComboboxEmpty>No results found.</ComboboxEmpty>
              {options.map((option) => (
                <ComboboxItem key={option.value} value={option.value}>
                  {option.label}
                </ComboboxItem>
              ))}
            </ComboboxContent>
          </Combobox>
          {description && <FormDescription>{description}</FormDescription>}
          <FormMessage />
        </FormItem>
      )}
    />
  );
}
