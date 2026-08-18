import { useState } from "react";
import { Plus, Trash2, MessageSquare } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { useToast } from "@/components/ui/toast";
import { useSessions, useCreateSession, useDeleteSession, type ChatSession } from "@/hooks/useChat";

interface Props {
  documentId: string;
  activeSessionId: string | null;
  onSelectSession: (session: ChatSession) => void;
}

export default function SessionList({ documentId, activeSessionId, onSelectSession }: Props) {
  const { data: sessions, isLoading } = useSessions(documentId);
  const createSession = useCreateSession();
  const deleteSession = useDeleteSession();
  const { toast } = useToast();
  const [deleteTarget, setDeleteTarget] = useState<ChatSession | null>(null);

  const handleCreate = async () => {
    const session = await createSession.mutateAsync(documentId);
    onSelectSession(session);
  };

  const handleDeleteConfirm = () => {
    if (!deleteTarget) return;
    deleteSession.mutate(deleteTarget.id, {
      onSuccess: () => toast("Сессия удалена", "success"),
      onError: (err) => toast(err.message, "error"),
    });
  };

  return (
    <div className="flex h-full min-h-0 flex-col border-r">
      <div className="flex items-center justify-between px-3 border-b h-12">
        <h3 className="text-sm font-medium">Сессии</h3>
        <Button
          size="icon"
          variant="ghost"
          className="h-7 w-7"
          title="Новая сессия"
          onClick={handleCreate}
          disabled={createSession.isPending}
        >
          <Plus className="h-4 w-4" />
        </Button>
      </div>
      <ScrollArea className="min-h-0 flex-1">
        <div className="p-2 space-y-1">
          {isLoading && (
            <p className="text-xs text-muted-foreground p-2">Загрузка...</p>
          )}
          {sessions?.length === 0 && !isLoading && (
            <p className="text-xs text-muted-foreground p-2">
              Нет сессий. Создайте новую.
            </p>
          )}
          {sessions?.map((session) => (
            <div
              key={session.id}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelectSession(session); } }}
              className={`group flex items-center gap-2 rounded-lg px-3 py-2 text-sm cursor-pointer transition-colors ${
                session.id === activeSessionId
                  ? "bg-accent text-accent-foreground"
                  : "hover:bg-accent/50"
              }`}
              onClick={() => onSelectSession(session)}
            >
              <MessageSquare className="h-4 w-4 shrink-0" />
              <span className="flex-1 truncate">
                {session.title || "Новая сессия"}
              </span>
              <Button
                size="icon"
                variant="ghost"
                className="h-6 w-6 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity"
                title="Удалить сессию"
                onClick={(e) => {
                  e.stopPropagation();
                  setDeleteTarget(session);
                }}
              >
                <Trash2 className="h-3 w-3" />
              </Button>
            </div>
          ))}
        </div>
      </ScrollArea>
      <ConfirmDialog
        open={!!deleteTarget}
        onOpenChange={(open) => { if (!open) setDeleteTarget(null); }}
        title="Удалить сессию?"
        description={`Сессия «${deleteTarget?.title || "Новая сессия"}» и вся история сообщений будут удалены.`}
        confirmLabel="Удалить"
        variant="destructive"
        onConfirm={handleDeleteConfirm}
      />
    </div>
  );
}
