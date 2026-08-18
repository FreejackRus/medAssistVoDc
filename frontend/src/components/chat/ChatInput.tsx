import { useRef, useState, useEffect, type FormEvent } from "react";
import { FileText, Paperclip, Send, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useUploadMaxBytes } from "@/hooks/useAppConfig";

interface Props {
  onSend: (message: string, files?: File[]) => void;
  disabled?: boolean;
}

const MAX_CHAT_ATTACHMENTS = 5;

function formatFileSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} КБ`;
  return `${(bytes / 1024 / 1024).toFixed(1)} МБ`;
}

export default function ChatInput({ onSend, disabled }: Props) {
  const [value, setValue] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [fileError, setFileError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const maxBytes = useUploadMaxBytes();
  const maxMB = Math.round(maxBytes / 1024 / 1024);

  // Warn before leaving with unsent text
  useEffect(() => {
    if (!value.trim() && files.length === 0) return;
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault();
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [value, files.length]);

  const addFiles = (selected: FileList | null) => {
    if (!selected || disabled) return;
    setFileError(null);
    const next = [...files];

    for (const file of Array.from(selected)) {
      if (next.length >= MAX_CHAT_ATTACHMENTS) {
        setFileError(`Можно приложить не больше ${MAX_CHAT_ATTACHMENTS} PDF`);
        break;
      }
      if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
        setFileError("В чат можно приложить только PDF");
        continue;
      }
      if (file.size > maxBytes) {
        setFileError(`Файл '${file.name}' больше лимита ${maxMB} МБ`);
        continue;
      }
      const duplicate = next.some(
        (item) => item.name === file.name && item.size === file.size && item.lastModified === file.lastModified,
      );
      if (!duplicate) next.push(file);
    }

    setFiles(next);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed, files);
    setValue("");
    setFiles([]);
    setFileError(null);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-2 border-t p-4">
      {files.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {files.map((file) => (
            <span
              key={`${file.name}-${file.size}-${file.lastModified}`}
              className="inline-flex max-w-full items-center gap-1.5 rounded-md border bg-muted/50 px-2 py-1 text-xs"
            >
              <FileText className="size-3.5 shrink-0" />
              <span className="max-w-48 truncate">{file.name}</span>
              <span className="shrink-0 text-muted-foreground">{formatFileSize(file.size)}</span>
              <button
                type="button"
                className="rounded p-0.5 hover:bg-muted"
                onClick={() => setFiles((current) => current.filter((item) => item !== file))}
                disabled={disabled}
                title="Убрать файл"
              >
                <X className="size-3" />
              </button>
            </span>
          ))}
        </div>
      )}
      {fileError && <p className="text-xs text-destructive">{fileError}</p>}
      <div className="flex gap-2">
        <input
          ref={fileInputRef}
          type="file"
          accept="application/pdf,.pdf"
          multiple
          className="hidden"
          onChange={(event) => addFiles(event.target.files)}
        />
        <Button
          type="button"
          variant="outline"
          size="icon"
          title="Прикрепить PDF"
          disabled={disabled}
          onClick={() => fileInputRef.current?.click()}
        >
          <Paperclip className="h-4 w-4" />
        </Button>
        <Input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Введите сообщение..."
          disabled={disabled}
          className="flex-1"
        />
        <Button type="submit" size="icon" title="Отправить" disabled={disabled || !value.trim()}>
          <Send className="h-4 w-4" />
        </Button>
      </div>
    </form>
  );
}
