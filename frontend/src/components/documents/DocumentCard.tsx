import { FileText, Trash2, Loader2, AlertCircle, CheckCircle2, RotateCw } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { Document } from "@/hooks/useDocuments";

const statusConfig: Record<string, { icon: typeof Loader2; label: string; variant: "default" | "secondary" | "destructive" }> = {
  processing: { icon: Loader2, label: "Обработка", variant: "secondary" },
  ready: { icon: CheckCircle2, label: "Готов", variant: "default" },
  error: { icon: AlertCircle, label: "Ошибка", variant: "destructive" },
};

interface Props {
  doc: Document;
  selected: boolean;
  onSelect: () => void;
  onDelete: () => void;
  onRetry: () => void;
  retrying?: boolean;
}

export default function DocumentCard({ doc, selected, onSelect, onDelete, onRetry, retrying }: Props) {
  const status = statusConfig[doc.status] ?? statusConfig.processing;
  const StatusIcon = status.icon;
  const canReprocess = doc.status !== "processing";
  const reprocessTitle =
    doc.status === "error" ? "Повторить обработку" : "Переобработать документ";

  return (
    <Card
      className={`cursor-pointer transition-colors hover:bg-accent/50 ${
        selected ? "border-primary bg-accent/40 ring-2 ring-inset ring-primary" : ""
      }`}
      onClick={onSelect}
    >
      <CardContent className="grid grid-cols-[auto_minmax(0,1fr)] gap-3 p-4 sm:grid-cols-[auto_minmax(0,1fr)_auto] sm:items-center">
        <FileText className="row-span-2 h-8 w-8 shrink-0 text-muted-foreground sm:row-span-1" />
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium">{doc.filename}</p>
          {doc.diagnosis_name && (
            <p className="truncate text-xs text-muted-foreground">
              {doc.diagnosis_name}
              {doc.mkb_code ? ` (${doc.mkb_code})` : ""}
            </p>
          )}
        </div>
        <div className="col-start-2 flex min-w-0 items-center gap-1 sm:col-start-auto">
          <Badge variant={status.variant} className="shrink-0 gap-1">
            <StatusIcon className={`h-3 w-3 ${doc.status === "processing" ? "animate-spin" : ""}`} />
            {status.label}
          </Badge>
          {canReprocess && (
            <Button
              variant="ghost"
              size="icon"
              className="shrink-0"
              title={reprocessTitle}
              disabled={retrying}
              onClick={(e) => {
                e.stopPropagation();
                onRetry();
              }}
            >
              {retrying ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RotateCw className="h-4 w-4" />
              )}
            </Button>
          )}
          <Button
            variant="ghost"
            size="icon"
            className="shrink-0"
            title="Удалить документ"
            onClick={(e) => {
              e.stopPropagation();
              onDelete();
            }}
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
