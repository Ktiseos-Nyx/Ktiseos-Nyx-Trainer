import {
  useCallback,
  useMemo,
  type ComponentProps,
  type ReactNode,
} from "react";
import { Command as CommandPrimitive } from "cmdk";
import { Check } from "lucide-react";

import {
  Popover,
  PopoverAnchor,
  PopoverContent,
} from "@/components/ui/popover";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupInput,
} from "@/components/ui/input-group";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

import { useControlledState } from "@/hooks/use-controlled-state";
import {
  useFilteredOptions,
  type SelectionOption,
} from "@/hooks/use-filtered-options";

type MultiSelectProps<T extends string> = {
  /** Full list of options to display. */
  items: SelectionOption<T>[];

  /** Controlled array of currently selected values. */
  value: T[];
  /** Called with the full updated selection array after each toggle. */
  onValueChange: (value: T[]) => void;

  /** Controlled open state of the dropdown. Omit to let the component manage it internally. */
  open?: boolean;
  /** Called whenever the open state changes. */
  onOpenChange?: (open: boolean) => void;

  /** Controlled search query. Omit to let the component manage query state internally. */
  query?: string;
  /** Called whenever the search query changes. */
  onQueryChange?: (query: string) => void;

  /**
   * Custom renderer for each list item.
   * Receives the option and a boolean indicating whether it is currently selected.
   * When omitted a default label + checkmark layout is used.
   */
  renderOption?: (option: SelectionOption<T>, selected: boolean) => ReactNode;

  /** Placeholder shown in the search input when empty. @default "Search..." */
  placeholder?: string;

  /** Content shown when no options match the current query. @default "No results found" */
  emptyContent?: ReactNode;
  /**
   * Content shown in place of the option list while `isLoading` is `true`.
   * Defaults to a single skeleton bar.
   */
  loadingContent?: ReactNode;

  /**
   * Custom filter predicate. Receives each option and the current query string.
   * Replaces the default case-insensitive label match when provided.
   */
  filterFn?: (item: SelectionOption<T>, query: string) => boolean;
  /**
   * Skip local filtering entirely.
   * Use when `items` are already filtered server-side for the current query.
   * @default false
   */
  disableLocalFilter?: boolean;

  /** Render the loading skeleton in place of the option list. */
  isLoading?: boolean;

  /** Clear the search query after each selection toggle. @default true */
  clearQueryOnSelect?: boolean;
  /** Close the dropdown immediately after each selection toggle. @default false */
  closeAfterSelect?: boolean;

  /** Content rendered as a leading addon inside the input (e.g. a search icon or label). */
  startAddon?: ReactNode;
  /** Content rendered as a trailing addon inside the input (e.g. a selected-count badge or clear button). */
  endAddon?: ReactNode;
} & ComponentProps<typeof PopoverContent>;

/**
 * Multi-select combobox that toggles values in and out of a selection array.
 *
 * **Required:** `value`, `onValueChange`, `items`
 *
 * Supports controlled `open` and `query` state, async loading, custom option rendering,
 * and custom filtering. Embed icons or actions inside the input via `startAddon` / `endAddon`.
 */
export function MultiSelect<T extends string>({
  items,
  value,
  onValueChange,
  open: openProp,
  onOpenChange,
  query: queryProp,
  onQueryChange,
  renderOption,
  placeholder = "Search...",
  emptyContent = "No results found",
  loadingContent,
  filterFn,
  disableLocalFilter = false,
  isLoading,
  clearQueryOnSelect = true,
  closeAfterSelect = false,
  startAddon,
  endAddon,
  ...props
}: MultiSelectProps<T>) {
  const [values, setValues] = useControlledState({
    value,
    defaultValue: [],
    onChange: onValueChange,
  });

  const [open, setOpen] = useControlledState({
    value: openProp,
    defaultValue: false,
    onChange: onOpenChange,
  });

  const [query, setQuery] = useControlledState({
    value: queryProp,
    defaultValue: "",
    onChange: onQueryChange,
  });

  const selectedSet = useMemo(() => new Set(values), [values]);

  const filteredItems = useFilteredOptions({
    items,
    query,
    filterFn,
    disableLocalFilter,
  });

  const toggleValue = useCallback(
    (nextValue: string) => {
      const v = nextValue as T;

      const next = selectedSet.has(v)
        ? values.filter((item) => item !== v)
        : [...values, v];

      setValues(next);

      if (clearQueryOnSelect) {
        setQuery("");
      }

      if (closeAfterSelect) {
        setOpen(false);
      }
    },
    [
      values,
      selectedSet,
      setValues,
      clearQueryOnSelect,
      setQuery,
      closeAfterSelect,
      setOpen,
    ],
  );

  return (
    <div data-slot="combobox-content" className="relative w-full">
      <Popover open={open} onOpenChange={setOpen}>
        <Command shouldFilter={false}>
          <PopoverAnchor asChild>
            <InputGroup>
              {startAddon && (
                <InputGroupAddon align="inline-start">
                  {startAddon}
                </InputGroupAddon>
              )}
              <CommandPrimitive.Input
                asChild
                value={query}
                onValueChange={setQuery}
                onMouseDown={() => setOpen(true)}
                onKeyDown={(e) => {
                  if (e.key === "Escape") {
                    if (open) {
                      setOpen(false);
                    } else {
                      setQuery("");
                    }
                    return;
                  }
                  setOpen(true);
                }}
              >
                <InputGroupInput
                  placeholder={placeholder}
                  onFocus={() => setOpen(true)}
                />
              </CommandPrimitive.Input>
              {endAddon && (
                <InputGroupAddon align="inline-end">{endAddon}</InputGroupAddon>
              )}
            </InputGroup>
          </PopoverAnchor>

          {!open && <CommandList aria-hidden="true" className="hidden" />}

          <PopoverContent
            asChild
            className="w-(--radix-popover-trigger-width) p-0"
            onInteractOutside={(event) => {
              if (
                event.target instanceof Element &&
                event.target.hasAttribute("cmdk-input")
              ) {
                event.preventDefault();
              }
            }}
            {...props}
          >
            <CommandList>
              {isLoading ? (
                <CommandPrimitive.Loading>
                  {loadingContent ?? (
                    <div className="p-2">
                      <Skeleton className="h-6 w-full" />
                    </div>
                  )}
                </CommandPrimitive.Loading>
              ) : filteredItems.length ? (
                <CommandGroup>
                  {filteredItems.map((item) => {
                    const isSelected = selectedSet.has(item.value);

                    return (
                      <CommandItem
                        key={item.value}
                        value={item.value}
                        disabled={item.disabled}
                        onSelect={toggleValue}
                        onMouseDown={(e) => e.preventDefault()}
                      >
                        {renderOption ? (
                          renderOption(item, isSelected)
                        ) : (
                          <>
                            <Check
                              className={cn(
                                "mr-2 h-4 w-4",
                                isSelected ? "opacity-100" : "opacity-0",
                              )}
                            />
                            {item.label}
                          </>
                        )}
                      </CommandItem>
                    );
                  })}
                </CommandGroup>
              ) : (
                <CommandEmpty>{emptyContent}</CommandEmpty>
              )}
            </CommandList>
          </PopoverContent>
        </Command>
      </Popover>
    </div>
  );
}
