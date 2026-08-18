import { useState } from "react";
import { Check, Copy, Loader2, Download, Play, RotateCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/components/ui/toast";
import MarkdownRenderer from "@/components/shared/MarkdownRenderer";
import {
  useGenerateAlgorithm,
  exportPdf,
  type AlgorithmGenerationMode,
} from "@/hooks/useAlgorithm";

interface Props {
  documentId: string;
  documentName: string;
}

const ALGORITHM_MODES: { value: AlgorithmGenerationMode; label: string; title: string }[] = [
  {
    value: "physician",
    label: "Врачебный",
    title: "Рабочий алгоритм с маршрутизацией и чек-листом",
  },
  {
    value: "structured",
    label: "Расширенный",
    title: "Расширенный практический алгоритм",
  },
  {
    value: "source",
    label: "По разделам",
    title: "Алгоритм по структуре исходного документа",
  },
];

export default function AlgorithmView({ documentId, documentName }: Props) {
  const [modeSelection, setModeSelection] = useState<{
    documentId: string;
    mode: AlgorithmGenerationMode;
  } | null>(null);
  const modeOverride =
    modeSelection?.documentId === documentId ? modeSelection.mode : null;
  const {
    content,
    isStreaming,
    isLoadingSaved,
    error,
    generate,
    hasSaved,
    isRunningSaved,
    activeMode,
  } = useGenerateAlgorithm(documentId, modeOverride);
  const { toast } = useToast();
  const [copied, setCopied] = useState(false);
  const isBusy = isLoadingSaved || isStreaming;

  const handleExport = async () => {
    try {
      await exportPdf(content, documentName);
    } catch (e) {
      toast(e instanceof Error ? e.message : "Ошибка экспорта", "error");
    }
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      toast("Не удалось скопировать текст", "error");
    }
  };

  return (
    <Card>
      <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-3 space-y-0 pb-3">
        <CardTitle className="min-w-0 flex-1 text-base">
          Алгоритм: {documentName}
        </CardTitle>
        <div className="flex w-full flex-wrap items-center justify-end gap-2 lg:w-auto">
          <div
            className="grid h-9 w-full grid-cols-3 rounded-md border bg-muted/40 p-0.5 sm:w-auto"
            role="group"
            aria-label="Формат алгоритма"
          >
            {ALGORITHM_MODES.map((option) => (
              <Button
                key={option.value}
                type="button"
                size="sm"
                variant={activeMode === option.value ? "secondary" : "ghost"}
                className="h-8 px-2.5 text-xs"
                aria-pressed={activeMode === option.value}
                title={option.title}
                disabled={isBusy}
                onClick={() =>
                  setModeSelection({ documentId, mode: option.value })
                }
              >
                {option.label}
              </Button>
            ))}
          </div>
          <Button
            size="sm"
            onClick={generate}
            disabled={isBusy}
            className="gap-2"
          >
            {isBusy ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : hasSaved ? (
              <RotateCw className="h-4 w-4" />
            ) : (
              <Play className="h-4 w-4" />
            )}
            {isLoadingSaved
              ? "Загрузка..."
              : isStreaming
              ? "Генерация..."
              : hasSaved
                ? "Перегенерировать"
                : "Сгенерировать"}
          </Button>
          {content && !isStreaming && (
            <>
              <Button
                size="sm"
                variant="outline"
                onClick={handleCopy}
                className="gap-2"
                title="Копировать текст"
              >
                {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                {copied ? "Скопировано" : "Копировать"}
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={handleExport}
                className="gap-2"
              >
                <Download className="h-4 w-4" />
                PDF
              </Button>
            </>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {error && (
          <div className="mb-4 flex items-center gap-3 rounded-lg bg-destructive/10 px-4 py-2.5 text-sm text-destructive">
            <p className="flex-1">{error}</p>
            <Button
              variant="outline"
              size="sm"
              className="shrink-0 gap-1.5"
              onClick={generate}
            >
              <RotateCw className="h-3.5 w-3.5" />
              Повторить
            </Button>
          </div>
        )}
        {isLoadingSaved && !content && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Загрузка...
          </div>
        )}
        {content ? (
          <MarkdownRenderer content={content} variant="algorithm" />
        ) : (
          !isStreaming &&
          !isLoadingSaved && (
            <p className="text-sm text-muted-foreground">
              Нажмите «Сгенерировать» для создания диагностического алгоритма
            </p>
          )
        )}
        {isStreaming && !content && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            {isRunningSaved ? "Генерация продолжается..." : "Подключение к AI..."}
          </div>
        )}
        {isRunningSaved && content && (
          <div className="mt-4 flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Генерация продолжается...
          </div>
        )}
      </CardContent>
    </Card>
  );
}
