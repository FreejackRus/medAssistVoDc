import { useState, useMemo } from "react";
import { FileText } from "lucide-react";
import { useDocuments, type Document } from "@/hooks/useDocuments";
import { useSessions, type ChatSession } from "@/hooks/useChat";
import SessionList from "@/components/chat/SessionList";
import ChatWindow from "@/components/chat/ChatWindow";
import { ScrollArea } from "@/components/ui/scroll-area";
import { QueryError } from "@/components/shared/QueryError";

export default function ChatPage() {
  const { data: docs, isLoading, error, refetch } = useDocuments();
  const readyDocs = useMemo(() => docs?.filter((d) => d.status === "ready") ?? [], [docs]);
  const [selectedDoc, setSelectedDoc] = useState<Document | null>(null);
  const [activeSession, setActiveSession] = useState<ChatSession | null>(null);

  // Clear selection if document was deleted
  const validDoc =
    selectedDoc && readyDocs.some((d) => d.id === selectedDoc.id) ? selectedDoc : null;

  // Clear session if document was deselected or session was deleted
  const { data: sessions } = useSessions(validDoc?.id);
  const validSession =
    validDoc && activeSession && sessions?.some((s) => s.id === activeSession.id)
      ? activeSession
      : null;

  return (
    <div className="flex h-full min-h-0 overflow-hidden">
      {/* Document selector */}
      <div className="flex min-h-0 w-56 shrink-0 flex-col border-r">
        <div className="flex items-center px-3 border-b h-12">
          <h3 className="text-sm font-medium">Документы</h3>
        </div>
        <ScrollArea className="min-h-0 flex-1">
          <div className="p-2 space-y-1">
            {isLoading && (
              <p className="text-xs text-muted-foreground p-2">Загрузка...</p>
            )}
            {error && (
              <QueryError
                error={error}
                onRetry={() => void refetch()}
                className="p-3 text-xs"
              />
            )}
            {!isLoading && !error && readyDocs.length === 0 && (
              <p className="text-xs text-muted-foreground p-2">
                Нет готовых документов
              </p>
            )}
            {readyDocs.map((doc) => (
              <button
                key={doc.id}
                type="button"
                className={`flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm text-left transition-colors ${
                  doc.id === validDoc?.id
                    ? "bg-accent text-accent-foreground"
                    : "hover:bg-accent/50"
                }`}
                onClick={() => {
                  setSelectedDoc(doc);
                  setActiveSession(null);
                }}
              >
                <FileText className="h-4 w-4 shrink-0" />
                <span className="truncate">
                  {doc.diagnosis_name ?? doc.filename}
                </span>
              </button>
            ))}
          </div>
        </ScrollArea>
      </div>

      {/* Session list */}
      {validDoc ? (
        <div className="min-h-0 w-56 shrink-0">
          <SessionList
            documentId={validDoc.id}
            activeSessionId={validSession?.id ?? null}
            onSelectSession={setActiveSession}
          />
        </div>
      ) : null}

      {/* Chat window */}
      <div className="min-h-0 min-w-0 flex-1">
        {validSession ? (
          <ChatWindow sessionId={validSession.id} />
        ) : (
          <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
            {validDoc
              ? "Выберите или создайте сессию чата"
              : "Выберите документ для начала диалога"}
          </div>
        )}
      </div>
    </div>
  );
}
