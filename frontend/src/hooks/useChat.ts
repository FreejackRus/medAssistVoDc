import { useState, useCallback, useRef, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiFetch, readResumeSSE, readUploadSSE } from "@/lib/api";
import { createTextSmoother, type TextSmoother } from "@/lib/textSmoother";

export interface ChatSession {
  id: string;
  user_id: string;
  document_id: string;
  title: string;
  created_at: string;
}

export interface ChatMessage {
  id: string;
  session_id: string;
  role: "user" | "assistant";
  content: string;
  status: "running" | "completed" | "error";
  stream_seq: number;
  created_at: string;
  attachments: ChatAttachment[];
}

export interface ChatAttachment {
  id: string;
  message_id: string;
  filename: string;
  mime_type: string;
  size_bytes: number;
  status: "ready" | "error";
  created_at: string;
}

export function useSessions(documentId?: string) {
  const params = documentId ? `?document_id=${documentId}` : "";
  return useQuery<ChatSession[]>({
    queryKey: ["sessions", documentId],
    queryFn: () => apiFetch(`/chat/sessions${params}`),
  });
}

export function useMessages(sessionId: string | null) {
  const qc = useQueryClient();
  const activeStreamsRef = useRef<
    Map<string, { controller: AbortController; smoother: TextSmoother }>
  >(new Map());
  const query = useQuery<ChatMessage[]>({
    queryKey: ["messages", sessionId],
    queryFn: () => apiFetch(`/chat/sessions/${sessionId}/messages`),
    enabled: !!sessionId,
  });

  const runningSignature =
    query.data
      ?.filter((m) => m.role === "assistant" && m.status === "running")
      .map((m) => `${m.id}:${m.stream_seq}`)
      .join("|") ?? "";

  useEffect(() => {
    if (!sessionId) return;

    const runningMessages =
      query.data?.filter((m) => m.role === "assistant" && m.status === "running") ?? [];
    const runningIds = new Set(runningMessages.map((m) => m.id));

    for (const [messageId, active] of activeStreamsRef.current) {
      if (!runningIds.has(messageId)) {
        active.controller.abort();
        active.smoother.stop();
        activeStreamsRef.current.delete(messageId);
      }
    }

    for (const message of runningMessages) {
      if (activeStreamsRef.current.has(message.id)) continue;

      const controller = new AbortController();
      let lastSeq = message.stream_seq ?? 0;

      const updateMessage = (patch: Partial<ChatMessage>) => {
        qc.setQueryData<ChatMessage[]>(["messages", sessionId], (current) =>
          current?.map((item) =>
            item.id === message.id ? { ...item, ...patch } : item,
          ) ?? current,
        );
      };
      const smoother = createTextSmoother({
        initialText: message.content,
        onText: (text) => updateMessage({ content: text }),
      });
      activeStreamsRef.current.set(message.id, { controller, smoother });

      const run = async () => {
        try {
          for await (const event of readResumeSSE(
            `/chat/messages/${message.id}/stream?after=${lastSeq}`,
            controller.signal,
          )) {
            if (event.seq && event.seq <= lastSeq) continue;
            if (event.seq) lastSeq = event.seq;

            if (event.event === "token") {
              smoother.enqueue(event.content);
            } else if (event.event === "error") {
              smoother.flush();
              updateMessage({ status: "error", content: smoother.getText() });
              break;
            } else if (event.event === "done") {
              smoother.flush();
              updateMessage({ status: "completed", content: smoother.getText() });
              break;
            }
          }
        } catch {
          if (!controller.signal.aborted) {
            smoother.flush();
            updateMessage({ status: "error", content: smoother.getText() });
          }
        } finally {
          activeStreamsRef.current.delete(message.id);
          if (!controller.signal.aborted) {
            smoother.flush();
            qc.invalidateQueries({ queryKey: ["messages", sessionId] });
            qc.invalidateQueries({ queryKey: ["sessions"] });
          } else {
            smoother.stop();
          }
        }
      };

      void run();
    }
  }, [sessionId, runningSignature, qc, query.data]);

  useEffect(
    () => () => {
      for (const active of activeStreamsRef.current.values()) {
        active.controller.abort();
        active.smoother.stop();
      }
      activeStreamsRef.current.clear();
    },
    [],
  );

  return query;
}

export function useCreateSession() {
  const qc = useQueryClient();
  return useMutation<ChatSession, Error, string>({
    mutationFn: (documentId) =>
      apiFetch("/chat/sessions", {
        method: "POST",
        body: JSON.stringify({ document_id: documentId }),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["sessions"] }),
  });
}

export function useDeleteSession() {
  const qc = useQueryClient();
  return useMutation<unknown, Error, string>({
    mutationFn: (id) =>
      apiFetch(`/chat/sessions/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["sessions"] }),
  });
}

export function useSendMessage(sessionId: string | null) {
  const qc = useQueryClient();
  const [streamContent, setStreamContent] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pendingMessage, setPendingMessage] = useState<string | null>(null);
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [lastFailedMessage, setLastFailedMessage] = useState<string | null>(null);
  const [lastFailedFiles, setLastFailedFiles] = useState<File[]>([]);
  const abortRef = useRef<AbortController | null>(null);
  const bufferRef = useRef("");
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    setStreamContent("");
    setIsStreaming(false);
    setError(null);
    setPendingMessage(null);
    setPendingFiles([]);
    setLastFailedMessage(null);
    setLastFailedFiles([]);
    return () => {
      abortRef.current?.abort();
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [sessionId]);

  const send = useCallback(
    async (message: string, files: File[] = []) => {
      if (!sessionId) return;
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setPendingMessage(message);
      setPendingFiles(files);
      bufferRef.current = "";
      setStreamContent("");
      setIsStreaming(true);
      setError(null);
      setLastFailedMessage(null);

      const flush = () => {
        rafRef.current = null;
        setStreamContent(bufferRef.current);
      };

      try {
        const formData = new FormData();
        formData.append("message", message);
        for (const file of files) {
          formData.append("attachments", file);
        }

        for await (const token of readUploadSSE(
          `/chat/sessions/${sessionId}/messages`,
          formData,
          controller.signal,
        )) {
          bufferRef.current += token;
          if (rafRef.current === null) {
            rafRef.current = requestAnimationFrame(flush);
          }
        }
      } catch (e) {
        if (controller.signal.aborted) return;
        setError(e instanceof Error ? e.message : "Ошибка отправки сообщения");
        setLastFailedMessage(message);
        setLastFailedFiles(files);
      } finally {
        if (rafRef.current !== null) {
          cancelAnimationFrame(rafRef.current);
          rafRef.current = null;
        }
        if (!controller.signal.aborted) {
          setStreamContent(bufferRef.current);
          setIsStreaming(false);
          setPendingMessage(null);
          setPendingFiles([]);
          qc.invalidateQueries({ queryKey: ["messages", sessionId] });
          qc.invalidateQueries({ queryKey: ["sessions"] });
        }
      }
    },
    [sessionId, qc],
  );

  const retry = useCallback(() => {
    if (lastFailedMessage) send(lastFailedMessage, lastFailedFiles);
  }, [lastFailedMessage, lastFailedFiles, send]);

  return { streamContent, isStreaming, error, send, pendingMessage, pendingFiles, retry, canRetry: !!lastFailedMessage };
}
