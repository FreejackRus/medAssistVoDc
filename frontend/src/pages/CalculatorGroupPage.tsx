import { Link, Navigate, useNavigate, useParams } from "react-router";
import { ArrowLeft, ChevronRight, Loader2 } from "lucide-react";
import CalculatorCard from "@/components/calculators/CalculatorCard";
import { Select } from "@/components/ui/select";
import { useCalculatorRegistry } from "@/hooks/useCalculators";
import { cn } from "@/lib/utils";
import { QueryError } from "@/components/shared/QueryError";

export default function CalculatorGroupPage() {
  const navigate = useNavigate();
  const { groupId, calculatorId } = useParams();
  const { data: registry, isLoading, error, refetch } = useCalculatorRegistry();
  const group = registry?.groups.find((item) => item.id === groupId);
  const groupCalculators = group?.calculators ?? [];
  const firstCalculator = groupCalculators[0];
  const activeCalculator = groupCalculators.find(
    (calculator) => calculator.id === calculatorId,
  );

  if (isLoading) {
    return (
      <div className="flex justify-center p-12">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 sm:p-6">
        <QueryError error={error} onRetry={() => void refetch()} />
      </div>
    );
  }

  if (!group || !firstCalculator) {
    return (
      <div className="p-4 sm:p-6">
        <div className="mx-auto max-w-3xl rounded-lg border bg-card p-6 text-center">
          <h2 className="text-lg font-semibold">Направление не найдено</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Возможно, оно было перемещено или удалено.
          </p>
          <Link
            to="/calculators"
            className="mt-4 inline-flex h-9 items-center gap-2 rounded-md border px-3 text-sm font-medium transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <ArrowLeft className="h-4 w-4" />
            К калькуляторам
          </Link>
        </div>
      </div>
    );
  }

  if (!activeCalculator) {
    return (
      <Navigate
        to={`/calculators/${group.id}/${firstCalculator.id}`}
        replace
      />
    );
  }

  return (
    <div className="p-4 sm:p-6">
      <div className="mx-auto max-w-6xl space-y-6">
        <div className="space-y-3">
          <Link
            to="/calculators"
            className="inline-flex items-center gap-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <ArrowLeft className="h-4 w-4" />
            Калькуляторы
          </Link>
          <div>
            <h2 className="text-2xl font-bold">{group.title}</h2>
            <p className="text-sm text-muted-foreground">
              Калькуляторов в разделе: {groupCalculators.length}
            </p>
          </div>
        </div>

        <div className="grid items-start gap-6 lg:grid-cols-[minmax(240px,300px)_minmax(0,680px)]">
          <aside className="hidden lg:block">
            <p className="mb-2 text-xs font-medium uppercase text-muted-foreground">
              Калькуляторы направления
            </p>
            <nav className="overflow-hidden rounded-lg border bg-card">
              {groupCalculators.map((calculator, index) => {
                const isActive = calculator.id === activeCalculator.id;
                return (
                  <Link
                    key={calculator.id}
                    to={`/calculators/${group.id}/${calculator.id}`}
                    aria-current={isActive ? "page" : undefined}
                    className={cn(
                      "flex min-h-14 items-center gap-3 px-3 py-2.5 text-sm transition-colors hover:bg-muted/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring",
                      index > 0 && "border-t",
                      isActive && "bg-muted font-medium text-foreground",
                    )}
                  >
                    <span className="min-w-0 flex-1">{calculator.title}</span>
                    <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground" />
                  </Link>
                );
              })}
            </nav>
          </aside>

          <div className="min-w-0 space-y-3">
            <div className="lg:hidden">
              <label className="mb-1.5 block text-sm font-medium" htmlFor="calculator-selector">
                Калькулятор
              </label>
              <Select
                id="calculator-selector"
                value={activeCalculator.id}
                options={groupCalculators.map((calculator) => ({
                  value: calculator.id,
                  label: calculator.title,
                }))}
                onValueChange={(nextCalculatorId) =>
                  navigate(`/calculators/${group.id}/${nextCalculatorId}`)
                }
              />
            </div>
            <CalculatorCard key={activeCalculator.id} config={activeCalculator} />
          </div>
        </div>
      </div>
    </div>
  );
}
