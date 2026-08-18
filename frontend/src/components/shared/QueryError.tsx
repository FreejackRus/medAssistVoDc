import { AlertCircle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";

interface QueryErrorProps {
  error: unknown;
  onRetry?: () => void;
  className?: string;
}

export function QueryError({ error, onRetry, className }: QueryErrorProps) {
  const message = error instanceof Error ? error.message : "Не удалось загрузить данные";
  return (
    <div
      className={`rounded-lg border border-destructive/30 bg-destructive/5 p-4 ${className ?? ""}`}
      role="alert"
    >
      <div className="flex items-start gap-3">
        <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-destructive" />
        <div className="min-w-0 flex-1">
          <p className="font-medium">Ошибка загрузки</p>
          <p className="mt-1 text-sm text-muted-foreground">{message}</p>
        </div>
        {onRetry && (
          <Button variant="outline" size="sm" className="gap-2" onClick={onRetry}>
            <RefreshCw className="h-4 w-4" />
            Повторить
          </Button>
        )}
      </div>
    </div>
  );
}
