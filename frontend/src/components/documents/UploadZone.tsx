import { useCallback, useRef, useState, type DragEvent } from "react";
import { Upload, Loader2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/toast";
import { useUploadDocument } from "@/hooks/useDocuments";
import { useUploadMaxBytes } from "@/hooks/useAppConfig";

interface Props {
  compact?: boolean;
}

export default function UploadZone({ compact = false }: Props) {
  const upload = useUploadDocument();
  const { toast } = useToast();
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const maxBytes = useUploadMaxBytes();
  const maxMB = Math.round(maxBytes / 1024 / 1024);

  const handleCancel = () => {
    abortRef.current?.abort();
    abortRef.current = null;
  };

  const handleFile = useCallback(
    (file: File) => {
      if (file.type !== "application/pdf") {
        toast("Поддерживаются только PDF файлы", "error");
        return;
      }
      if (file.size > maxBytes) {
        toast(`Файл слишком большой (макс. ${maxMB} МБ)`, "error");
        return;
      }
      abortRef.current = new AbortController();
      upload.mutate(
        { file, signal: abortRef.current.signal },
        {
          onSuccess: (doc) =>
            toast(`Документ «${doc.filename}» загружен`, "success"),
          onError: (err) => {
            if (err.name === "AbortError") {
              toast("Загрузка отменена", "info");
            } else {
              toast(err.message, "error");
            }
          },
          onSettled: () => { abortRef.current = null; },
        },
      );
    },
    [upload, toast, maxBytes, maxMB],
  );

  const onDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
      className={`flex flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed transition-colors ${
        compact ? "p-4" : "p-8"
      } ${
        dragOver ? "border-primary bg-primary/5" : "border-muted-foreground/25"
      }`}
    >
      <Upload className={`${compact ? "h-5 w-5" : "h-8 w-8"} text-muted-foreground`} />
      <div className="text-center">
        <p className="text-sm text-muted-foreground">
          Перетащите PDF или нажмите для выбора
        </p>
        <p className="text-xs text-muted-foreground/60 mt-1">
          Формат: PDF, максимум {maxMB} МБ
        </p>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept=".pdf"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
          e.target.value = "";
        }}
      />
      <div className="flex gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => inputRef.current?.click()}
          disabled={upload.isPending}
        >
          {upload.isPending ? (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          ) : null}
          {upload.isPending ? "Загрузка..." : "Выбрать файл"}
        </Button>
        {upload.isPending && (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCancel}
            className="gap-1.5 text-destructive hover:text-destructive"
          >
            <X className="h-4 w-4" />
            Отмена
          </Button>
        )}
      </div>
    </div>
  );
}
