import { useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router";
import {
  Activity,
  Baby,
  Brain,
  ChevronRight,
  Droplets,
  FlaskConical,
  HeartPulse,
  LayoutGrid,
  List,
  Loader2,
  Search,
  Stethoscope,
  Ruler,
  Wind,
} from "lucide-react";
import { useCalculatorRegistry } from "@/hooks/useCalculators";
import CalculatorCard from "@/components/calculators/CalculatorCard";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { QueryError } from "@/components/shared/QueryError";

type ViewMode = "groups" | "all";

const groupIcons: Record<string, typeof Ruler> = {
  "general-practice": Stethoscope,
  cardiology: HeartPulse,
  nephrology: Activity,
  pulmonology: Wind,
  "emergency-icu": Brain,
  hepatology: FlaskConical,
  endocrinology: Droplets,
  hematology: Droplets,
  obstetrics: Baby,
  pediatrics: Baby,
};

function formatCalculatorCount(count: number) {
  const lastTwoDigits = count % 100;
  const lastDigit = count % 10;

  if (lastTwoDigits >= 11 && lastTwoDigits <= 14) return `${count} калькуляторов`;
  if (lastDigit === 1) return `${count} калькулятор`;
  if (lastDigit >= 2 && lastDigit <= 4) return `${count} калькулятора`;
  return `${count} калькуляторов`;
}

export default function CalculatorsPage() {
  const { data: registry, isLoading, error, refetch } = useCalculatorRegistry();
  const [searchParams, setSearchParams] = useSearchParams();
  const [query, setQuery] = useState("");
  const viewMode: ViewMode = searchParams.get("view") === "all" ? "all" : "groups";
  const normalizedQuery = query.trim().toLocaleLowerCase("ru");
  const groups = useMemo(
    () =>
      (registry?.groups ?? [])
        .map((group) => ({
          ...group,
          calculators: group.calculators.filter((calculator) =>
            `${calculator.title} ${calculator.description}`
              .toLocaleLowerCase("ru")
              .includes(normalizedQuery),
          ),
        }))
        .filter((group) => group.calculators.length > 0),
    [normalizedQuery, registry?.groups],
  );
  const calculators = groups.flatMap((group) => group.calculators);

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

      <div className="relative max-w-xl">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          aria-label="Поиск медицинских калькуляторов"
          className="pl-9"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Поиск по названию или показателю..."
        />
      </div>

      {isLoading && (
        <div className="flex justify-center py-12">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      )}
      {error && <QueryError error={error} onRetry={() => void refetch()} />}

      {!isLoading && !error && calculators.length === 0 && (
        <p className="py-8 text-center text-sm text-muted-foreground">
          Калькуляторы не найдены.
        </p>
      )}

      {!isLoading && !error && calculators.length > 0 && (
        <>
      {viewMode === "groups" ? (
        <div className="overflow-hidden rounded-lg border bg-card">
          {groups.map((group, index) => {
            const groupCalculators = group.calculators;
            const firstCalculator = groupCalculators[0];
            const GroupIcon = groupIcons[group.id] ?? Stethoscope;

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
                    {formatCalculatorCount(groupCalculators.length)}
                  </span>
                </span>
                <span className="hidden shrink-0 text-sm text-muted-foreground sm:block">
                  Открыть раздел
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
        </>
      )}
    </div>
  );
}
