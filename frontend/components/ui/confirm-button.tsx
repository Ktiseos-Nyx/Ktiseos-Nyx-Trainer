import * as React from "react"

import { cn } from "@/lib/utils"

interface ConfirmButtonProps
  extends Omit<React.ComponentPropsWithoutRef<"button">, "onClick"> {
  /** Called only on the second (confirming) click. */
  onConfirm: () => void
  /** Label shown after the first click while awaiting confirmation. */
  confirmText?: string
  /** How long (ms) the armed state stays before reverting to idle. */
  timeout?: number
}

export function ConfirmButton({
  onConfirm,
  confirmText = "Confirm?",
  timeout = 3000,
  children,
  className,
  disabled,
  ...props
}: ConfirmButtonProps) {
  const [armed, setArmed] = React.useState(false)

  React.useEffect(() => {
    if (!armed) return
    const id = window.setTimeout(() => setArmed(false), timeout)
    return () => window.clearTimeout(id)
  }, [armed, timeout])

  function onClick() {
    if (armed) {
      setArmed(false)
      onConfirm()
    } else {
      setArmed(true)
    }
  }

  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      onBlur={() => setArmed(false)}
      data-armed={armed || undefined}
      className={cn(
        "inline-flex h-9 items-center justify-center rounded-md px-4 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 disabled:pointer-events-none disabled:opacity-50",
        armed
          ? "bg-destructive text-destructive-foreground hover:bg-destructive/90 focus-visible:ring-destructive"
          : "border border-input bg-background hover:bg-accent hover:text-accent-foreground focus-visible:ring-ring",
        className
      )}
      {...props}
    >
      <span aria-live="polite" className="sr-only">
        {armed ? "Press again to confirm" : ""}
      </span>
      {armed ? confirmText : children}
    </button>
  )
}
