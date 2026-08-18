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
  return (
    <div className={cn("relative", className)}>
      <select
        id={id}
        data-slot="select"
        value={value}
        disabled={disabled}
        className={cn(
          "h-9 w-full appearance-none rounded-md border border-input bg-background px-3 py-1 pr-9 text-left text-sm shadow-sm transition-colors outline-none hover:border-foreground/30 focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 disabled:pointer-events-none disabled:cursor-not-allowed disabled:bg-input/50 disabled:opacity-60",
          triggerClassName
        )}
        onChange={(event) => onValueChange(event.target.value)}
      >
        {!options.some((option) => option.value === value) && (
          <option value="" disabled>{placeholder}</option>
        )}
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      <ChevronDown
        aria-hidden="true"
        className="pointer-events-none absolute right-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
      />
    </div>
  )
}

export { Select, type SelectOption }
