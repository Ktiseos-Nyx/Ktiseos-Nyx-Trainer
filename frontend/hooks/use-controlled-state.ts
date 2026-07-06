import { useCallback, useState } from "react";

type UseControlledStateProps<T> = {
  /**
   * Controlled value. When provided the component is in controlled mode and
   * internal state is ignored. Pass `undefined` for uncontrolled mode.
   */
  value?: T;
  /** Initial value used when the component is uncontrolled. */
  defaultValue: T;
  /** Called whenever the value changes, regardless of controlled/uncontrolled mode. */
  onChange?: (value: T) => void;
};

/**
 * Bridges controlled and uncontrolled state in a single hook.
 *
 * Returns `[state, setState]` like `useState`. When `value` is provided the
 * hook operates in controlled mode — `setState` calls `onChange` but does not
 * touch internal state. When `value` is `undefined` the hook manages its own
 * state and still calls `onChange` on every change.
 *
 * @example
 * // Uncontrolled (component owns the state)
 * const [open, setOpen] = useControlledState({ defaultValue: false });
 *
 * // Controlled (parent owns the state)
 * const [open, setOpen] = useControlledState({ value: props.open, defaultValue: false, onChange: props.onOpenChange });
 */
export function useControlledState<T>({
  value,
  defaultValue,
  onChange,
}: UseControlledStateProps<T>) {
  const [internalValue, setInternalValue] = useState<T>(defaultValue);

  const isControlled = value !== undefined;
  const state = isControlled ? value : internalValue;
  const setState = useCallback(
    (next: T) => {
      if (!isControlled) {
        setInternalValue(next);
      }
      onChange?.(next);
    },
    [isControlled, onChange],
  );
  return [state, setState] as const;
}
