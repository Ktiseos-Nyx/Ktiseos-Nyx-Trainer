import { useCallback, useEffect, useRef, useState } from "react";

export type AsyncStatus = "idle" | "pending" | "success" | "error";

/**
 * Discriminated union for every possible state of an async operation.
 * Narrow on `status` (or the boolean flags) to get typed `data` / `error` access.
 */
export type AsyncState<TData, TError> =
  | {
      status: "idle";
      data: undefined;
      error: undefined;
      isIdle: true;
      isLoading: false;
      isSuccess: false;
      isError: false;
    }
  | {
      status: "pending";
      data: TData | undefined; // preserves previous data while refetching
      error: undefined;
      isIdle: false;
      isLoading: true;
      isSuccess: false;
      isError: false;
    }
  | {
      status: "success";
      data: TData;
      error: undefined;
      isIdle: false;
      isLoading: false;
      isSuccess: true;
      isError: false;
    }
  | {
      status: "error";
      data: undefined;
      error: TError;
      isIdle: false;
      isLoading: false;
      isSuccess: false;
      isError: true;
    };

export type UseAsyncOptions<TData, TError, TArgs extends unknown[]> = {
  /** The async function to run. Receives the same args passed to `execute`. */
  action: (...args: TArgs) => Promise<TData> | TData;
  /** Called with the resolved value and original args on success. */
  onSuccess?: (data: TData, args: TArgs) => void;
  /** Called with the thrown error and original args on failure. */
  onError?: (error: TError, args: TArgs) => void;
  /** Called after every settled attempt regardless of outcome. */
  onSettled?: (
    data: TData | undefined,
    error: TError | undefined,
    args: TArgs,
  ) => void;
};

export type UseAsyncReturn<TData, TError, TArgs extends unknown[]> = AsyncState<
  TData,
  TError
> & {
  /**
   * Run the action and await the result , even though side effect can be dispatch with onError ,
   * but it will **Reject on failure** for imperative error handling
   * Use when the next step depends on the  outcome (sequencing , guarded , navigation) or your own
   * try/catch , Mirrors Tanstack `mutateAsync`
   */
  executeAsync: (...args: TArgs) => Promise<TData>;
  /**
   * Fire and Forget , **It never reject** , error handling can be listen declaratively with
   * sideEfect with onError
   * Use when the action is the end of the story (e.g a button that just shows success/error).
   * Mirror Tanstack `mutate`
   */
  execute: (...args: TArgs) => void;
  reset: () => void;
};

const IDLE_STATE = {
  status: "idle",
  data: undefined,
  error: undefined,
  isIdle: true,
  isLoading: false,
  isSuccess: false,
  isError: false,
} as const;

/**
 * Wraps an async function with loading, success, and error state.
 *
 * - Stale responses are dropped — only the most-recent `execute` call can commit
 *   state. Calling `execute` again while in-flight silently cancels the earlier one.
 * - Callbacks are always fresh; no stale-closure risk without wrapping in `useCallback`.
 * - `reset` cancels any in-flight request and returns to the `idle` state.
 */
export function useAsync<
  TData = unknown,
  TError = unknown,
  TArgs extends unknown[] = [],
>(
  options: UseAsyncOptions<TData, TError, TArgs>,
): UseAsyncReturn<TData, TError, TArgs> {
  const [state, setState] = useState<AsyncState<TData, TError>>(IDLE_STATE);

  // always fresh callbacks without forcing execute to change identity (deps array)
  const optionsRef = useRef(options);
  useEffect(() => {
    optionsRef.current = options;
  });

  // latest wins , stale response will dropped
  const requestIdRef = useRef(0);

  // prevent setState after mounted
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const executeAsync = useCallback(async (...args: TArgs): Promise<TData> => {
    // Tăng trước khi đọc
    const requestId = ++requestIdRef.current;
    const { action, onSuccess, onError, onSettled } = optionsRef.current;
    setState((prev) => ({
      status: "pending",
      data: prev.data,
      error: undefined,
      isIdle: false,
      isLoading: true,
      isSuccess: false,
      isError: false,
    }));
    try {
      const data = await action(...args);
      if (requestId === requestIdRef.current && mountedRef.current) {
        // commit state only on the latest && component still mounted
        setState({
          status: "success",
          data,
          error: undefined,
          isIdle: false,
          isLoading: false,
          isSuccess: true,
          isError: false,
        });
        onSuccess?.(data, args);
        onSettled?.(data, undefined, args);
      }

      return data;
    } catch (err) {
      const error = err as TError;
      if (requestId === requestIdRef.current && mountedRef.current) {
        setState({
          status: "error",
          data: undefined,
          error,
          isIdle: false,
          isLoading: false,
          isSuccess: false,
          isError: true,
        });
        onError?.(error, args);
        onSettled?.(undefined, error, args);
      }
      throw err;
    }
  }, []);

  const execute = useCallback(async (...args: TArgs) => {
    // Fire-and-forget: errors are already committed to state and passed to
    // onError inside executeAsync, so we swallow the rejection here to spare
    // callers from having to .catch. This is what makes execute "never reject".
    try {
      const data = await executeAsync(...args);
      return data;
    } catch {}
  }, []);

  const reset = useCallback(() => {
    requestIdRef.current++;
    if (mountedRef.current) {
      setState(IDLE_STATE);
    }
  }, []);

  return { ...state, execute, executeAsync, reset };
}
