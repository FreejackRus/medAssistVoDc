import * as React from "react"
import { Check } from "lucide-react"

import { cn } from "@/lib/utils"

function Checkbox({
  className,
  disabled,
  ...props
}: Omit<React.ComponentProps<"input">, "type">) {
  return (
    <span
      data-slot="checkbox"
      className={cn(
        "relative inline-flex size-5 shrink-0 items-center justify-center align-middle",
        disabled && "cursor-not-allowed opacity-60",
        className
      )}
    >
      <input
        type="checkbox"
        disabled={disabled}
        className="peer sr-only"
        {...props}
      />
      <span
        aria-hidden="true"
        className="flex size-5 items-center justify-center rounded-md border border-foreground/65 bg-background text-transparent shadow-xs transition-colors peer-checked:border-primary peer-checked:bg-primary peer-checked:text-primary-foreground peer-focus-visible:border-ring peer-focus-visible:ring-3 peer-focus-visible:ring-ring/50 peer-disabled:border-input peer-disabled:bg-input/50 peer-disabled:shadow-none peer-enabled:hover:border-foreground peer-enabled:hover:bg-muted/40"
      >
        <Check className="size-3.5 stroke-[3]" />
      </span>
    </span>
  )
}

export { Checkbox }
