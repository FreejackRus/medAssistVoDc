import { useState } from "react";
import { User, Bot, Copy, Check, Loader2, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";
import MarkdownRenderer from "@/components/shared/MarkdownRenderer";
import type { ChatMessage } from "@/hooks/useChat";

function relativeTime(dateStr: string): string {
  const now = Date.now();
  const then = new Date(dateStr + "Z").getTime(); // SQLite dates are UTC
  const diff = Math.floor((now - then) / 1000);
  if (diff < 60) return "только что";
  if (diff < 3600) return `${Math.floor(diff / 60)} мин. назад`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} ч. назад`;
  return new Date(dateStr + "Z").toLocaleDateString("ru-RU", {
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

interface Props {
  message: ChatMessage;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} КБ`;
  return `${(bytes / 1024 / 1024).toFixed(1)} МБ`;
}

function AttachmentList({ attachments, inverted = false }: { attachments: ChatMessage["attachments"]; inverted?: boolean }) {
  if (!attachments.length) return null;
  return (
    <div className="mt-2 flex flex-wrap gap-1.5">
      {attachments.map((attachment) => (
        <span
          key={attachment.id}
          className={`inline-flex max-w-full items-center gap-1.5 rounded-md border px-2 py-1 text-xs ${
            inverted ? "border-primary-foreground/30 bg-primary-foreground/10" : "bg-background/70"
          }`}
        >
          <FileText className="size-3.5 shrink-0" />
          <span className="max-w-52 truncate">{attachment.filename}</span>
          <span className={inverted ? "text-primary-foreground/70" : "text-muted-foreground"}>
            {formatFileSize(attachment.size_bytes)}
          </span>
        </span>
      ))}
    </div>
  );
}

export default function MessageBubble({ message }: Props) {
  const isUser = message.role === "user";
  const isRunning = message.status === "running";
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const fullDate = new Date(message.created_at + "Z").toLocaleString("ru-RU");

  return (
    <div className={`group flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      <div
        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${
          isUser ? "bg-primary text-primary-foreground" : "bg-muted"
        }`}
      >
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>
      <div className={`max-w-[75%] ${isUser ? "text-right" : ""}`}>
        <div
          className={`relative rounded-2xl px-4 py-2.5 ${
            isUser
              ? "bg-primary text-primary-foreground"
              : "bg-muted"
          }`}
        >
          {isUser ? (
            <>
              <p className="text-sm whitespace-pre-wrap">{message.content}</p>
              <AttachmentList attachments={message.attachments} inverted />
            </>
          ) : isRunning && !message.content ? (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Думаю...
            </div>
          ) : (
            <>
              <MarkdownRenderer content={message.content} />
              {isRunning && (
                <div className="mt-2 flex items-center gap-1.5 text-xs text-muted-foreground">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  Ответ дописывается...
                </div>
              )}
            </>
          )}
          {message.content && (
            <Button
              variant="ghost"
              size="icon"
              className={`absolute -top-2 z-10 h-7 w-7 border bg-background shadow-sm opacity-0 transition-opacity group-hover:opacity-100 ${
                isUser ? "-left-2" : "-right-2"
              }`}
              title="Копировать"
              onClick={handleCopy}
            >
              {copied ? (
                <Check className="h-3.5 w-3.5 text-green-600" />
              ) : (
                <Copy className="h-3.5 w-3.5" />
              )}
            </Button>
          )}
        </div>
        <p
          className="mt-0.5 text-[10px] text-muted-foreground/60 opacity-0 group-hover:opacity-100 transition-opacity"
          title={fullDate}
        >
          {relativeTime(message.created_at)}
        </p>
      </div>
    </div>
  );
}
