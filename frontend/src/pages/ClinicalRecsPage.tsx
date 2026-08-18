import { useState, useMemo } from "react";
import { Loader2, Search, ChevronLeft, ChevronRight } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useClinicalRecs } from "@/hooks/useClinicalRecs";
import { useDebouncedValue } from "@/hooks/useDebouncedValue";
import RecommendationCard from "@/components/clinical-recs/RecommendationCard";

const PAGE_SIZE = 20;

export default function ClinicalRecsPage() {
  const { data: recs, isLoading, error } = useClinicalRecs();
  const [search, setSearch] = useState("");
  const debouncedSearch = useDebouncedValue(search, 250);
  const [page, setPage] = useState(0);

  const filtered = useMemo(() => {
    if (!recs) return [];
    if (!debouncedSearch.trim()) return recs;
    const q = debouncedSearch.toLowerCase();
    return recs.filter(
      (r) =>
        r.title.toLowerCase().includes(q) ||
        r.mkb_code.toLowerCase().includes(q) ||
        r.keywords.toLowerCase().includes(q),
    );
  }, [recs, debouncedSearch]);

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const paged = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

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
          placeholder="Поиск по названию, МКБ-коду или ключевым словам..."
          className="pl-9"
          value={search}
          onChange={(e) => {
            setSearch(e.target.value);
            setPage(0);
          }}
        />
      </div>

      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      )}

      {error && (
        <p className="text-sm text-destructive">
          Ошибка загрузки: {error.message}
        </p>
      )}

      {!isLoading && filtered.length > 0 && (
        <p className="text-xs text-muted-foreground">
          {search
            ? `Найдено: ${filtered.length}`
            : `Всего: ${filtered.length}`}
        </p>
      )}

      {!isLoading && filtered.length === 0 && (
        <p className="text-center text-sm text-muted-foreground py-8">
          {search ? "Ничего не найдено" : "Нет клинических рекомендаций"}
        </p>
      )}

      <div className="space-y-2">
        {paged.map((rec) => (
          <RecommendationCard key={rec.id} rec={rec} />
        ))}
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-3">
          <Button
            variant="outline"
            size="sm"
            className="gap-1"
            disabled={page === 0}
            onClick={() => setPage((p) => p - 1)}
          >
            <ChevronLeft className="h-4 w-4" />
            Назад
          </Button>
          <span className="text-sm text-muted-foreground tabular-nums">
            {page + 1} / {totalPages}
          </span>
          <Button
            variant="outline"
            size="sm"
            className="gap-1"
            disabled={page >= totalPages - 1}
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
