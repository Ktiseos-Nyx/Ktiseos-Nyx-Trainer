import { Button } from "@/components/ui/button";
import {
  useAsync,
  type UseAsyncOptions,
} from "@/hooks/use-async";
import { type ComponentProps, type ReactNode } from "react";
import { Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

export type AsyncButtonProps<
  TData,
  TError,
  TArgs extends unknown[] = [],
> = Omit<ComponentProps<typeof Button>, "onClick"> &
  UseAsyncOptions<TData, TError, TArgs> & {
    /** Content displayed while the action is in flight. Defaults to `children` when omitted. */
    loadingText?: ReactNode;
  };

/**
 * A `Button` that manages its own async state.
 *
 * Disables itself while the action is in flight and swaps children for a spinner
 * (+ optional `loadingText`) via an invisible overlay — the button width never shifts.
 *
 * **Required:** `action`
 */
export function AsyncButton<TData, TError, TArgs extends unknown[] = []>({
  action,
  loadingText,
  onSuccess,
  onError,
  onSettled,
  children,
  disabled,
  className,
  ...props
}: AsyncButtonProps<TData, TError, TArgs>) {
  const { isLoading, execute } = useAsync({
    action,
    onSuccess,
    onError,
    onSettled,
  });

  const resolvedLoadingText = loadingText ?? children;

  return (
    <Button
      {...props}
      disabled={disabled || isLoading}
      onClick={() => {
        (execute as () => Promise<TData>)();
      }}
      className={cn("grid grid-cols-1 place-items-center", className)}
    >
      <div
        className={cn(
          "col-start-1 row-start-1 flex items-center justify-center gap-2 transition-all",
          isLoading ? "invisible opacity-0" : "visible opacity-100",
        )}
      >
        {children}
      </div>

      <div
        className={cn(
          "col-start-1 row-start-1 flex items-center justify-center gap-2",
          isLoading ? "visible opacity-100" : "invisible opacity-0",
        )}
      >
        <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
        {resolvedLoadingText && <span>{resolvedLoadingText}</span>}
      </div>
    </Button>
  );
}
