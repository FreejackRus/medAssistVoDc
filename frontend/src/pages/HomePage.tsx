import { useState, useCallback } from "react";
import { Search, FileUp, BrainCircuit, MessageSquare } from "lucide-react";
import { useDocuments, type Document } from "@/hooks/useDocuments";
import { useDebouncedValue } from "@/hooks/useDebouncedValue";
import { Input } from "@/components/ui/input";
import UploadZone from "@/components/documents/UploadZone";
import DocumentList from "@/components/documents/DocumentList";
import AlgorithmView from "@/components/algorithm/AlgorithmView";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { QueryError } from "@/components/shared/QueryError";

export default function HomePage() {
  const { data: docs, error, refetch } = useDocuments();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const debouncedSearch = useDebouncedValue(search, 250);

  const hasDocs = !!docs?.length;

  const validSelected = selectedId
    ? (docs?.find((d) => d.id === selectedId) ?? null)
    : null;
  const activeDoc = validSelected ?? docs?.[0] ?? null;
  const readyCount = docs?.filter((doc) => doc.status === "ready").length ?? 0;

  const handleSelect = useCallback((doc: Document) => setSelectedId(doc.id), []);

  return (
    <div className={hasDocs ? "flex h-full flex-col gap-6 overflow-hidden p-6" : "space-y-6 p-6"}>
      <div>
        <h2 className="text-2xl font-bold">Документы</h2>
        <p className="text-sm text-muted-foreground">
          Загрузите клиническую рекомендацию для генерации алгоритма
        </p>
      </div>

      {error ? (
        <QueryError error={error} onRetry={() => void refetch()} />
      ) : !hasDocs ? (
        <>
          <UploadZone />

          <div className="space-y-4 rounded-lg border border-dashed p-6 text-center">
            <p className="text-sm font-medium">Как начать работу</p>
            <div className="flex flex-col items-center justify-center gap-6 text-xs text-muted-foreground sm:flex-row">
              <div className="flex w-40 flex-col items-center gap-1.5">
                <FileUp className="h-6 w-6" />
                <span>1. Загрузите PDF</span>
              </div>
              <span className="hidden sm:block">&#8594;</span>
              <div className="flex w-40 flex-col items-center gap-1.5">
                <BrainCircuit className="h-6 w-6" />
                <span>2. Сгенерируйте алгоритм</span>
              </div>
              <span className="hidden sm:block">&#8594;</span>
              <div className="flex w-40 flex-col items-center gap-1.5">
                <MessageSquare className="h-6 w-6" />
                <span>3. Задавайте вопросы в чате</span>
              </div>
            </div>
          </div>
        </>
      ) : (
        <div className="grid gap-4 xl:min-h-0 xl:flex-1 xl:grid-cols-[minmax(320px,420px)_minmax(0,1fr)] xl:overflow-hidden">
          <aside className="space-y-4 xl:flex xl:min-h-0 xl:flex-col xl:overflow-hidden">
            <UploadZone compact />

            <Card className="xl:flex xl:min-h-0 xl:flex-1 xl:flex-col">
              <CardHeader className="space-y-3 pb-3">
                <div className="flex items-center justify-between gap-3">
                  <CardTitle className="text-base">Клинические рекомендации</CardTitle>
                  <Badge variant="secondary">
                    {readyCount}/{docs?.length ?? 0}
                  </Badge>
                </div>
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    placeholder="Поиск по документам..."
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    className="pl-9"
                  />
                </div>
              </CardHeader>
              <CardContent className="pt-0 xl:min-h-0 xl:flex-1">
                <div className="max-h-[30vh] overflow-y-auto pr-1 sm:max-h-[50vh] xl:h-full xl:max-h-none">
                  <DocumentList
                    selectedId={activeDoc?.id ?? null}
                    onSelect={handleSelect}
                    search={debouncedSearch}
                  />
                </div>
              </CardContent>
            </Card>
          </aside>

          <section className="min-w-0 space-y-4 xl:flex xl:min-h-0 xl:flex-col xl:overflow-hidden">
            {activeDoc && (
              <div className="flex flex-col gap-2 rounded-lg border bg-muted/30 px-4 py-3 sm:flex-row sm:items-center sm:justify-between xl:shrink-0">
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium">
                    {activeDoc.diagnosis_name ?? activeDoc.filename}
                  </p>
                  <p className="truncate text-xs text-muted-foreground">
                    {activeDoc.filename}
                    {activeDoc.mkb_code ? ` · ${activeDoc.mkb_code}` : ""}
                  </p>
                </div>
                <Badge variant={activeDoc.status === "error" ? "destructive" : activeDoc.status === "ready" ? "default" : "secondary"}>
                  {documentStatusLabel(activeDoc.status)}
                </Badge>
              </div>
            )}

            <div className="xl:min-h-0 xl:flex-1 xl:overflow-y-auto xl:pr-1">
              {activeDoc?.status === "ready" ? (
                <AlgorithmView
                  key={activeDoc.id}
                  documentId={activeDoc.id}
                  documentName={activeDoc.diagnosis_name ?? activeDoc.filename}
                />
              ) : (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">
                      {activeDoc ? `Алгоритм: ${activeDoc.diagnosis_name ?? activeDoc.filename}` : "Алгоритм"}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      {activeDoc?.status === "processing"
                        ? "Документ еще обрабатывается. Алгоритм станет доступен после завершения обработки."
                        : "Выберите готовый документ слева, чтобы открыть или сгенерировать алгоритм."}
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

function documentStatusLabel(status: string) {
  switch (status) {
    case "ready":
      return "Готов";
    case "error":
      return "Ошибка";
    default:
      return "Обработка";
  }
}
