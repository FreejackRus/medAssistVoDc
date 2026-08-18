import { useEffect, useRef } from "react";
import { Loader2, Bot, User, RotateCw, FileText } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import MessageBubble from "./MessageBubble";
import ChatInput from "./ChatInput";
import MarkdownRenderer from "@/components/shared/MarkdownRenderer";
import { useMessages, useSendMessage, type ChatMessage } from "@/hooks/useChat";
import { QueryError } from "@/components/shared/QueryError";

interface Props {
  sessionId: string;
}

export default function ChatWindow({ sessionId }: Props) {
  const { data: messages, isLoading, error: queryError, refetch } = useMessages(sessionId);
  const { streamContent, isStreaming, error, send, pendingMessage, pendingFiles, retry, canRetry } = useSendMessage(sessionId);
  const hasPersistedRunning = messages?.some((msg) => msg.status === "running") ?? false;
  const showLocalPending = !!pendingMessage && !hasPersistedRunning;
  const showLocalStream = isStreaming && !hasPersistedRunning;
  const bottomRef = useRef<HTMLDivElement>(null);
  const rafRef = useRef(0);

  useEffect(() => {
    cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    });
  }, [messages, streamContent, pendingMessage]);

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden">
      <ScrollArea className="min-h-0 flex-1">
        <div className="space-y-4 p-4 pr-6">
          {isLoading && (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            </div>
          )}
          {queryError && (
            <QueryError error={queryError} onRetry={() => void refetch()} />
          )}
          {!isLoading && !queryError && messages?.length === 0 && !showLocalPending && (
            <div className="py-8 text-center">
              <p className="text-sm text-muted-foreground mb-4">
                Задайте вопрос по загруженному документу, например:
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {[
                  "Какой основной диагноз описан в документе?",
                  "Какие методы диагностики рекомендованы?",
                  "Опиши критерии постановки диагноза",
                  "Какие группы препаратов рекомендованы для лечения?",
                ].map((q) => (
                  <button
                    key={q}
                    className="rounded-full border px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
                    onClick={() => send(q)}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages?.map((msg: ChatMessage) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}
          {/* Show user's message immediately while waiting for response */}
          {showLocalPending && (
            <div className="flex gap-3 flex-row-reverse">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
                <User className="h-4 w-4" />
              </div>
              <div className="max-w-[75%] rounded-2xl bg-primary text-primary-foreground px-4 py-2.5">
                <p className="text-sm whitespace-pre-wrap">{pendingMessage}</p>
                {pendingFiles.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {pendingFiles.map((file) => (
                      <span
                        key={`${file.name}-${file.size}-${file.lastModified}`}
                        className="inline-flex max-w-full items-center gap-1.5 rounded-md border border-primary-foreground/30 bg-primary-foreground/10 px-2 py-1 text-xs"
                      >
                        <FileText className="size-3.5 shrink-0" />
                        <span className="max-w-52 truncate">{file.name}</span>
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
          {showLocalStream && streamContent && (
            <div className="flex gap-3">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-muted">
                <Bot className="h-4 w-4" />
              </div>
              <div className="max-w-[75%] rounded-2xl bg-muted px-4 py-2.5">
                <MarkdownRenderer content={streamContent} />
              </div>
            </div>
          )}
          {error && (
            <div className="mx-auto max-w-md rounded-lg bg-destructive/10 px-4 py-3 text-center text-sm text-destructive">
              <p>{error}</p>
              {canRetry && (
                <Button
                  variant="outline"
                  size="sm"
                  className="mt-2 gap-1.5"
                  onClick={retry}
                >
                  <RotateCw className="h-3.5 w-3.5" />
                  Повторить
                </Button>
              )}
            </div>
          )}
          {showLocalStream && !streamContent && (
            <div className="flex gap-3">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-muted">
                <Bot className="h-4 w-4" />
              </div>
              <div className="flex items-center gap-2 rounded-2xl bg-muted px-4 py-2.5 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                Думаю...
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </ScrollArea>
      <ChatInput onSend={send} disabled={isStreaming || hasPersistedRunning} />
    </div>
  );
}
