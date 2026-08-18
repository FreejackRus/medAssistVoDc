import * as React from "react"
import { ChevronDown } from "lucide-react"

import { cn } from "@/lib/utils"

type SelectOption = {
  value: string
  label: string
}

type SelectProps = {
  id?: string
  value: string
  options: SelectOption[]
  onValueChange: (value: string) => void
  placeholder?: string
  disabled?: boolean
  className?: string
  triggerClassName?: string
}

function Select({
  id,
  value,
  options,
  onValueChange,
  placeholder = "Выберите значение",
  disabled,
  className,
  triggerClassName,
}: SelectProps) {
  const [open, setOpen] = React.useState(false)
  const rootRef = React.useRef<HTMLDivElement>(null)
  const selected = options.find((option) => option.value === value)

  React.useEffect(() => {
    if (!open) return

    const closeOnOutsideClick = (event: MouseEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) {
        setOpen(false)
      }
    }

    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false)
      }
    }

    document.addEventListener("mousedown", closeOnOutsideClick)
    document.addEventListener("keydown", closeOnEscape)

    return () => {
      document.removeEventListener("mousedown", closeOnOutsideClick)
      document.removeEventListener("keydown", closeOnEscape)
    }
  }, [open])

  return (
    <div ref={rootRef} className={cn("relative", className)}>
      <button
        id={id}
        type="button"
        data-slot="select"
        aria-haspopup="listbox"
        aria-expanded={open}
        disabled={disabled}
        className={cn(
          "flex h-9 w-full items-center justify-between gap-2 rounded-md border border-input bg-background px-3 py-1 text-left text-sm shadow-sm transition-colors outline-none hover:border-foreground/30 focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 disabled:pointer-events-none disabled:cursor-not-allowed disabled:bg-input/50 disabled:opacity-60",
          triggerClassName
        )}
        onClick={() => setOpen((current) => !current)}
      >
        <span className={cn("min-w-0 flex-1 truncate", !selected && "text-muted-foreground")}>
          {selected?.label ?? placeholder}
        </span>
        <ChevronDown
          className={cn(
            "size-4 shrink-0 text-muted-foreground transition-transform",
            open && "rotate-180"
          )}
        />
      </button>

      {open && (
        <div
          data-slot="select-content"
          className="absolute left-0 right-0 top-full z-50 mt-1 max-h-60 overflow-auto rounded-md border border-border bg-popover p-1 text-popover-foreground shadow-lg"
        >
          <div role="listbox" aria-activedescendant={selected?.value}>
            {options.map((option) => {
              const isSelected = option.value === value
              return (
                <button
                  key={option.value}
                  type="button"
                  role="option"
                  aria-selected={isSelected}
                  className={cn(
                    "flex h-8 w-full items-center rounded-sm px-2.5 text-left text-sm transition-colors hover:bg-muted focus-visible:bg-muted focus-visible:outline-none",
                    isSelected && "bg-muted font-medium text-foreground"
                  )}
                  onClick={() => {
                    onValueChange(option.value)
                    setOpen(false)
                  }}
                >
                  <span className="truncate">{option.label}</span>
                </button>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

export { Select, type SelectOption }
