import { useState } from "react";
import { Loader2, Search, ChevronLeft, ChevronRight } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useClinicalRecs } from "@/hooks/useClinicalRecs";
import { useDebouncedValue } from "@/hooks/useDebouncedValue";
import RecommendationCard from "@/components/clinical-recs/RecommendationCard";
import { QueryError } from "@/components/shared/QueryError";

const PAGE_SIZE = 20;

export default function ClinicalRecsPage() {
  const [search, setSearch] = useState("");
  const debouncedSearch = useDebouncedValue(search, 250);
  const [page, setPage] = useState(1);
  const { data, isLoading, isFetching, error, refetch } = useClinicalRecs(
    debouncedSearch.trim(),
    page,
    PAGE_SIZE,
  );
  const recs = data?.recommendations ?? [];
  const totalPages = data?.total_pages ?? 0;

  return (
    <div className="p-6 space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Клинические рекомендации</h2>
        <p className="text-sm text-muted-foreground">
          Реестр Минздрава РФ
        </p>
      </div>

      <div className="relative max-w-md">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          aria-label="Поиск клинических рекомендаций"
          placeholder="Поиск по названию, МКБ-коду или ключевым словам..."
          className="pl-9"
          value={search}
          onChange={(e) => {
            setSearch(e.target.value);
            setPage(1);
          }}
        />
        {isFetching && !isLoading && (
          <Loader2
            aria-label="Обновление результатов"
            className="absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 animate-spin text-muted-foreground"
          />
        )}
      </div>

      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      )}

      {error && (
        <QueryError error={error} onRetry={() => void refetch()} />
      )}

      {!isLoading && !error && recs.length > 0 && (
        <p className="text-xs text-muted-foreground">
          {debouncedSearch.trim()
            ? `Найдено: ${data?.total ?? 0}`
            : `Всего: ${data?.total ?? 0}`}
        </p>
      )}

      {!isLoading && !error && recs.length === 0 && (
        <p className="text-center text-sm text-muted-foreground py-8">
          {search ? "Ничего не найдено" : "Нет клинических рекомендаций"}
        </p>
      )}

      <div className="space-y-2">
        {recs.map((rec) => (
          <RecommendationCard key={rec.id} rec={rec} />
        ))}
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-3">
          <Button
            variant="outline"
            size="sm"
            className="gap-1"
            disabled={page <= 1 || isFetching}
            onClick={() => setPage((p) => p - 1)}
          >
            <ChevronLeft className="h-4 w-4" />
            Назад
          </Button>
          <span className="text-sm text-muted-foreground tabular-nums">
            {page} / {totalPages}
          </span>
          <Button
            variant="outline"
            size="sm"
            className="gap-1"
            disabled={page >= totalPages || isFetching}
            onClick={() => setPage((p) => p + 1)}
          >
            Вперёд
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      )}
    </div>
  );
}
