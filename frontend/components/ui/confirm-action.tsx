import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogMedia,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { type ComponentProps, type ReactNode } from "react";
import { Button } from "@/components/ui/button";
import {
  AsyncButton,
  type AsyncButtonProps,
} from "@/components/ui/async-button";
import { useControlledState } from "@/hooks/use-controlled-state";

interface ConfirmActionProps<TData = unknown, TError = unknown> extends Omit<
  AsyncButtonProps<TData, TError, []>,
  "title"
> {
  title?: ReactNode;
  description?: ReactNode;
  cancelText?: ReactNode;
  confirmText?: ReactNode;
  confirmVariant?: ComponentProps<typeof Button>["variant"];
  size?: ComponentProps<typeof AlertDialogContent>["size"];
  media?: ReactNode;
  /** Controlled open state. Provide alongside `onOpenChange` to manage the dialog externally. */
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  /** Trigger element. Omit when controlling `open` externally. */
  children?: ReactNode;
}

/**
 * Guards inreversible actions behind a confirm dialog
 *
 * **Required:** `action`
 */
export function ConfirmAction<TData = unknown, TError = unknown>({
  action,
  title = "Are you sure?",
  description = "This action cannot be undone.",
  cancelText = "Cancel",
  confirmText = "Continue",
  confirmVariant = "default",
  size,
  media,
  loadingText,
  onSuccess,
  onError,
  onSettled,
  open: openProp,
  onOpenChange,
  children,
  ...props
}: ConfirmActionProps<TData, TError>) {
  const [open, setOpen] = useControlledState({
    value: openProp,
    defaultValue: false,
    onChange: onOpenChange,
  });

  return (
    <AlertDialog open={open} onOpenChange={setOpen}>
      {children && <AlertDialogTrigger asChild>{children}</AlertDialogTrigger>}

      <AlertDialogContent size={size}>
        <AlertDialogHeader>
          {media && <AlertDialogMedia>{media}</AlertDialogMedia>}
          <AlertDialogTitle>{title}</AlertDialogTitle>
          <AlertDialogDescription>{description}</AlertDialogDescription>
        </AlertDialogHeader>

        <AlertDialogFooter>
          <AlertDialogCancel>{cancelText}</AlertDialogCancel>
          <AsyncButton
            variant={confirmVariant}
            action={action}
            loadingText={loadingText ?? confirmText}
            onSuccess={(data, args) => {
              onSuccess?.(data, args);
              setOpen(false);
            }}
            onError={onError}
            onSettled={onSettled}
            {...props}
          >
            {confirmText}
          </AsyncButton>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
