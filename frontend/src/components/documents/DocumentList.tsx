import { useMemo, useState, useCallback, useRef, useEffect } from "react";
import { useDocuments, useDeleteDocument, useRetryDocument, type Document } from "@/hooks/useDocuments";
import { useToast } from "@/components/ui/toast";
import DocumentCard from "./DocumentCard";
import { Loader2 } from "lucide-react";

interface Props {
  selectedId: string | null;
  onSelect: (doc: Document) => void;
  search?: string;
}

export default function DocumentList({ selectedId, onSelect, search }: Props) {
  const { data: docs, isLoading } = useDocuments();
  const deleteMutation = useDeleteDocument();
  const retryMutation = useRetryDocument();
  const { toast } = useToast();
  const [pendingDeletes, setPendingDeletes] = useState<Set<string>>(new Set());
  const timersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());

  useEffect(() => {
    const timers = timersRef.current;
    return () => {
      timers.forEach((t) => clearTimeout(t));
      timers.clear();
    };
  }, []);

  const filtered = useMemo(() => {
    if (!docs) return [];
    const visible = docs.filter((d) => !pendingDeletes.has(d.id));
    if (!search?.trim()) return visible;
    const q = search.toLowerCase();
    return visible.filter(
      (d) =>
        d.filename.toLowerCase().includes(q) ||
        d.diagnosis_name?.toLowerCase().includes(q) ||
        d.mkb_code?.toLowerCase().includes(q),
    );
  }, [docs, search, pendingDeletes]);

  const handleDelete = useCallback((doc: Document) => {
    setPendingDeletes((prev) => new Set(prev).add(doc.id));

    const timer = setTimeout(() => {
      timersRef.current.delete(doc.id);
      deleteMutation.mutate(doc.id, {
        onSuccess: () => setPendingDeletes((prev) => {
          const next = new Set(prev);
          next.delete(doc.id);
          return next;
        }),
        onError: (err) => {
          setPendingDeletes((prev) => {
            const next = new Set(prev);
            next.delete(doc.id);
            return next;
          });
          toast(err.message, "error");
        },
      });
    }, 5000);

    timersRef.current.set(doc.id, timer);

    toast(`Документ «${doc.filename}» удалён`, "success", {
      label: "Отменить",
      onClick: () => {
        const t = timersRef.current.get(doc.id);
        if (t) {
          clearTimeout(t);
          timersRef.current.delete(doc.id);
        }
        setPendingDeletes((prev) => {
          const next = new Set(prev);
          next.delete(doc.id);
          return next;
        });
      },
    });
  }, [deleteMutation, toast]);

  const handleRetry = (doc: Document) => {
    retryMutation.mutate(doc.id, {
      onSuccess: () =>
        toast(
          doc.status === "error"
            ? `Повторная обработка «${doc.filename}» запущена`
            : `Переобработка «${doc.filename}» запущена`,
          "info",
        ),
      onError: (err) => toast(err.message, "error"),
    });
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!docs?.length) {
    return (
      <p className="py-4 text-center text-sm text-muted-foreground">
        Документы не загружены
      </p>
    );
  }

  if (filtered.length === 0) {
    return (
      <p className="py-4 text-center text-sm text-muted-foreground">
        Ничего не найдено
      </p>
    );
  }

  return (
    <div className="space-y-2">
      {filtered.map((doc) => (
        <DocumentCard
          key={doc.id}
          doc={doc}
          selected={doc.id === selectedId}
          onSelect={() => onSelect(doc)}
          onDelete={() => handleDelete(doc)}
          onRetry={() => handleRetry(doc)}
          retrying={retryMutation.isPending && retryMutation.variables === doc.id}
        />
      ))}
    </div>
  );
}
