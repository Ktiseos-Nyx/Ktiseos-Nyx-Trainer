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

import { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { UseFormReturn, FieldValues, Path, useWatch } from 'react-hook-form';
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
  SelectGroup,
  SelectItem,
  SelectLabel,
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
  label?: string;
  description?: string;
  placeholder?: string;
  disabled?: boolean;
  readOnly?: boolean;
}

type FieldOption = {
  value: string;
  label: string;
  description?: string;
};

type OptionGroup = {
  group: string;
  options: FieldOption[];
};

type SelectOptions = FieldOption[] | OptionGroup[];

function isGrouped(options: SelectOptions): options is OptionGroup[] {
  return options.length > 0 && 'group' in options[0];
}

interface SelectFormFieldProps<T extends FieldValues> extends BaseFieldProps<T> {
  options: SelectOptions;
}

interface ComboboxFormFieldProps<T extends FieldValues> extends BaseFieldProps<T> {
  options: FieldOption[];
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
          {label && <FormLabel>{label}</FormLabel>}
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
          {label && <FormLabel>{label}</FormLabel>}
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
 * Renders a form-connected select dropdown with label, optional description, and validation message.
 *
 * Binds the select value to the provided form control and synchronizes changes back to the form.
 *
 * @param form - React Hook Form instance controlling the field
 * @param name - Field name within the form
 * @param label - Visible label text for the select
 * @param description - Optional help text displayed below the control
 * @param options - Array of options to render; each option's `value` is used as the select value and `label` as the display text
 * @returns The rendered select form field element
 */
export function SelectFormField<T extends FieldValues>({
  form, name, label, description, options
}: SelectFormFieldProps<T>) {
  return (  // 👈 Return starts immediately
    <FormField
      control={form.control}
      name={name}
      render={({ field }) => (
        <FormItem>
          {label && <FormLabel>{label}</FormLabel>}
          <Select
            value={field.value}
            onValueChange={(val: string) => {
              field.onChange(val);
              form.setValue(name, val as any, { shouldDirty: true });
            }}
          >
            <FormControl><SelectTrigger><SelectValue /></SelectTrigger></FormControl>
            <SelectContent>
              {isGrouped(options)
                ? options.map((grp) => (
                    <SelectGroup key={grp.group}>
                      <SelectLabel>{grp.group}</SelectLabel>
                      {grp.options.map((opt) => (
                        <SelectItem key={opt.value} value={opt.value}>{opt.label}</SelectItem>
                      ))}
                    </SelectGroup>
                  ))
                : options.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value}>{opt.label}</SelectItem>
                  ))
              }
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
            {label && <FormLabel>{label}</FormLabel>}
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
          {label && <FormLabel>{label}</FormLabel>}
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
            {label && <FormLabel>{label}</FormLabel>}
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
 * Render a form-connected combobox that lets users pick an option or enter a custom value.
 *
 * Key design decisions to avoid race conditions:
 * - useWatch (not polling) for reactive form value tracking
 * - Only passes known option values to diceui's `value` prop (custom paths
 *   use '' so diceui doesn't fight with the input on blur)
 * - isTyping tracked via ref (not state) to avoid render-loop with useEffect
 */
export function ComboboxFormField<T extends FieldValues>({
  form, name, label, description, options, placeholder
}: ComboboxFormFieldProps<T>) {

  // Build a value→label lookup so we display filenames, not full paths
  const valueToLabel = useMemo(
    () => Object.fromEntries(options.map((o) => [o.value, o.label])),
    [options]
  );

  // useWatch: reactive, race-free observation of form value
  const formValue = (useWatch({ control: form.control, name }) ?? '') as string;

  // displayText = what the input shows (human-readable label or typed path)
  const [displayText, setDisplayText] = useState('');

  // Ref (not state) — avoids re-render cycles with the sync useEffect
  const isTypingRef = useRef(false);

  // Sync form value or options → displayText when NOT actively typing.
  // Covers: hydration, preset loads, programmatic setValue, options loading.
  useEffect(() => {
    if (!isTypingRef.current) {
      setDisplayText(valueToLabel[formValue] ?? formValue);
    }
  }, [formValue, valueToLabel]);

  // Filter by label when typing; show all options otherwise
  const filteredOptions = useMemo(() => {
    if (!isTypingRef.current || !displayText) return options;
    const lower = displayText.toLowerCase();
    return options.filter((o) => o.label.toLowerCase().includes(lower));
  }, [options, displayText]);

  // Only pass a value that matches a real option to diceui's value prop.
  // Custom paths pass '' so diceui doesn't try to sync/clear the input.
  const comboboxValue = useMemo(() => {
    return options.some(o => o.value === formValue) ? formValue : '';
  }, [formValue, options]);

  // Commit whatever is in the input to the form
  const commitInput = useCallback(() => {
    if (!displayText) {
      form.setValue(name, '' as any, { shouldDirty: true, shouldTouch: true });
      isTypingRef.current = false;
      return;
    }
    // If typed text matches an option's label, use that option's value
    const matchedOption = options.find(
      (o) => o.label.toLowerCase() === displayText.toLowerCase()
    );
    if (matchedOption) {
      form.setValue(name, matchedOption.value as any, { shouldDirty: true, shouldTouch: true });
      setDisplayText(matchedOption.label);
    } else {
      // Custom path — commit the raw text
      form.setValue(name, displayText as any, { shouldDirty: true, shouldTouch: true });
    }
    isTypingRef.current = false;
  }, [displayText, options, form, name]);

  return (
    <FormField
      control={form.control}
      name={name}
      render={() => (
        <FormItem>
          {label && <FormLabel>{label}</FormLabel>}
          <Combobox
            value={comboboxValue}
            onValueChange={(val: string) => {
              // User picked an item from the dropdown
              form.setValue(name, val as any, { shouldDirty: true, shouldTouch: true });
              setDisplayText(valueToLabel[val] ?? val);
              isTypingRef.current = false;
            }}
            inputValue={displayText}
            onInputValueChange={(val: string) => {
              setDisplayText(val);
              isTypingRef.current = true;
            }}
            onOpenChange={(open: boolean) => {
              if (!open && isTypingRef.current) {
                // Dropdown closed while user was typing — commit the input
                commitInput();
              }
            }}
            preserveInputOnBlur={true}
            manualFiltering={true}
          >
            <ComboboxAnchor>
              <FormControl><ComboboxInput placeholder={placeholder} /></FormControl>
              <ComboboxTrigger />
            </ComboboxAnchor>
            <ComboboxContent>
              <ComboboxEmpty>No matches — custom path will be used</ComboboxEmpty>
              {filteredOptions.map((opt) => (
                <ComboboxItem key={opt.value} value={opt.value}>{opt.label}</ComboboxItem>
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
