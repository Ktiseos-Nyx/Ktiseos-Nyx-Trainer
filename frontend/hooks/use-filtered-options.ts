import { useMemo } from "react";

export type SelectionOption<T extends string> = {
  /** Unique identifier for this option, used as the cmdk `value`. */
  value: T;
  /** Display string shown in the list and the search input. */
  label: string;
  /** When `true`, the item renders but cannot be selected. @default false */
  disabled?: boolean;
};

type UseFilteredOptionsProps<T extends string> = {
  items: SelectionOption<T>[];
  query: string;
  disableLocalFilter?: boolean;
  filterFn?: (item: SelectionOption<T>, query: string) => boolean;
};

/**
 * Filters a `SelectionOption` list against a search query.
 *
 * Returns the full list when `disableLocalFilter` is `true` or `query` is blank.
 * Applies `filterFn` when provided; otherwise uses a case-insensitive substring
 * match on `label`.
 */
export function useFilteredOptions<T extends string>({
  items,
  query,
  disableLocalFilter = false,
  filterFn,
}: UseFilteredOptionsProps<T>) {
  return useMemo(() => {
    if (disableLocalFilter || !query.trim()) {
      return items;
    }
    if (filterFn) {
      return items.filter((item) => filterFn(item, query));
    }
    const normalizedQuery = query.toLowerCase();
    return items.filter((item) =>
      item.label.toLowerCase().includes(normalizedQuery),
    );
  }, [items, query, disableLocalFilter, filterFn]);
}
