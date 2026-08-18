import { Link, useSearchParams } from "react-router";
import {
  Activity,
  ChevronRight,
  HeartPulse,
  LayoutGrid,
  List,
  Pill,
  Ruler,
} from "lucide-react";
import {
  calculatorGroups,
  calculators,
  type CalculatorGroupId,
} from "@/hooks/useCalculators";
import CalculatorCard from "@/components/calculators/CalculatorCard";
import { Button } from "@/components/ui/button";

type ViewMode = "groups" | "all";

const groupIcons = {
  anthropometry: Ruler,
  "renal-function": Activity,
  "cardiovascular-risk": HeartPulse,
  "medication-dosing": Pill,
} satisfies Record<CalculatorGroupId, typeof Ruler>;

function formatCalculatorCount(count: number) {
  const lastTwoDigits = count % 100;
  const lastDigit = count % 10;

  if (lastTwoDigits >= 11 && lastTwoDigits <= 14) return `${count} калькуляторов`;
  if (lastDigit === 1) return `${count} калькулятор`;
  if (lastDigit >= 2 && lastDigit <= 4) return `${count} калькулятора`;
  return `${count} калькуляторов`;
}

export default function CalculatorsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const viewMode: ViewMode = searchParams.get("view") === "all" ? "all" : "groups";

  const setViewMode = (mode: ViewMode) => {
    const nextParams = new URLSearchParams(searchParams);
    if (mode === "groups") {
      nextParams.delete("view");
    } else {
      nextParams.set("view", mode);
    }
    setSearchParams(nextParams, { replace: true });
  };

  return (
    <div className="space-y-6 p-4 sm:p-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 className="text-2xl font-bold">Медицинские калькуляторы</h2>
          <p className="text-sm text-muted-foreground">Расчёт клинических показателей</p>
        </div>

        <div className="inline-flex h-10 w-fit rounded-md border bg-background p-1">
          <Button
            type="button"
            variant={viewMode === "groups" ? "secondary" : "ghost"}
            size="sm"
            className="h-8 gap-2 px-3"
            aria-pressed={viewMode === "groups"}
            onClick={() => setViewMode("groups")}
          >
            <List className="h-4 w-4" />
            По направлениям
          </Button>
          <Button
            type="button"
            variant={viewMode === "all" ? "secondary" : "ghost"}
            size="sm"
            className="h-8 gap-2 px-3"
            aria-pressed={viewMode === "all"}
            onClick={() => setViewMode("all")}
          >
            <LayoutGrid className="h-4 w-4" />
            Все калькуляторы
          </Button>
        </div>
      </div>

      {viewMode === "groups" ? (
        <div className="overflow-hidden rounded-lg border bg-card">
          {calculatorGroups.map((group, index) => {
            const groupCalculators = calculators.filter(
              (calculator) => calculator.groupId === group.id,
            );
            const firstCalculator = groupCalculators[0];
            const GroupIcon = groupIcons[group.id];

            return (
              <Link
                key={group.id}
                to={
                  firstCalculator
                    ? `/calculators/${group.id}/${firstCalculator.id}`
                    : `/calculators/${group.id}`
                }
                className={`group flex min-h-20 items-center gap-4 px-4 py-3 transition-colors hover:bg-muted/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring sm:px-5 ${
                  index > 0 ? "border-t" : ""
                }`}
              >
                <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md border bg-background text-muted-foreground transition-colors group-hover:text-foreground">
                  <GroupIcon className="h-5 w-5" />
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block font-medium">{group.title}</span>
                  <span className="mt-0.5 block text-sm text-muted-foreground">
                    {group.description}
                  </span>
                </span>
                <span className="hidden shrink-0 text-sm text-muted-foreground sm:block">
                  {formatCalculatorCount(groupCalculators.length)}
                </span>
                <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground transition-transform group-hover:translate-x-0.5 group-hover:text-foreground" />
              </Link>
            );
          })}
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {calculators.map((calculator) => (
            <CalculatorCard key={calculator.id} config={calculator} />
          ))}
        </div>
      )}
    </div>
  );
}
