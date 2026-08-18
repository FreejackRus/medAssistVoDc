import { useState, useCallback, useRef, useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { apiBlob, apiFetch, readResumeSSE, readSSE } from "@/lib/api";
import { createTextSmoother } from "@/lib/textSmoother";

export interface Algorithm {
  id: string;
  document_id: string;
  content_markdown: string;
  status: "running" | "completed" | "error";
  generation_mode: AlgorithmGenerationMode;
  stream_seq: number;
  created_at: string;
}

export type AlgorithmGenerationMode = "structured" | "source" | "physician";

export function useAlgorithm(algoId: string | null) {
  return useQuery<Algorithm>({
    queryKey: ["algorithm", algoId],
    queryFn: () => apiFetch(`/algorithms/${algoId}`),
    enabled: !!algoId,
  });
}

export function useDocumentAlgorithm(documentId: string) {
  return useQuery<Algorithm | null>({
    queryKey: ["algorithm-by-doc", documentId],
    queryFn: () => apiFetch(`/algorithms/by-document/${documentId}`),
  });
}

export function useGenerateAlgorithm(
  documentId: string,
  modeOverride: AlgorithmGenerationMode | null = null,
) {
  const qc = useQueryClient();
  const { data: saved, isLoading: isLoadingSaved } = useDocumentAlgorithm(documentId);
  const activeMode = modeOverride ?? saved?.generation_mode ?? "physician";
  const [streamContent, setStreamContent] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const runningRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
  const resumeAbortRef = useRef<AbortController | null>(null);
  const resumeKeyRef = useRef<string | null>(null);
  const bufferRef = useRef("");
  const rafRef = useRef<number | null>(null);

  useEffect(
    () => () => {
      abortRef.current?.abort();
      resumeAbortRef.current?.abort();
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    },
    [],
  );

  const savedIsRunning = saved?.status === "running";
  const savedId = saved?.id;
  const savedStreamSeq = saved?.stream_seq ?? 0;
  const savedContent = saved?.content_markdown ?? "";
  const content = streamContent || savedContent;

  useEffect(() => {
    if (!savedId || !savedIsRunning || runningRef.current) return;

    const resumeKey = `${savedId}:${savedStreamSeq}`;
    if (resumeKeyRef.current === resumeKey) return;

    resumeAbortRef.current?.abort();
    const controller = new AbortController();
    resumeAbortRef.current = controller;
    resumeKeyRef.current = resumeKey;

    bufferRef.current = savedContent;
    setStreamContent(bufferRef.current);
    setIsStreaming(true);
    setError(null);

    let lastSeq = savedStreamSeq;
    const smoother = createTextSmoother({
      initialText: bufferRef.current,
      onText: (text) => {
        bufferRef.current = text;
        setStreamContent(text);
      },
    });

    const run = async () => {
      try {
        for await (const event of readResumeSSE(
          `/algorithms/${savedId}/stream?after=${lastSeq}`,
          controller.signal,
        )) {
          if (event.seq && event.seq <= lastSeq) continue;
          if (event.seq) lastSeq = event.seq;

          if (event.event === "token") {
            smoother.enqueue(event.content);
          } else if (event.event === "error") {
            throw new Error(event.content || "Ошибка генерации");
          } else if (event.event === "done") {
            smoother.flush();
            break;
          }
        }
      } catch (e) {
        if (controller.signal.aborted) return;
        setError(e instanceof Error ? e.message : "Ошибка генерации");
      } finally {
        if (rafRef.current !== null) {
          cancelAnimationFrame(rafRef.current);
          rafRef.current = null;
        }
        if (!controller.signal.aborted) {
          smoother.flush();
          bufferRef.current = smoother.getText();
          setStreamContent(bufferRef.current);
          setIsStreaming(false);
          resumeKeyRef.current = null;
          qc.invalidateQueries({ queryKey: ["algorithm-by-doc", documentId] });
        } else {
          smoother.stop();
        }
      }
    };

    void run();

    return () => {
      controller.abort();
      smoother.stop();
    };
  }, [documentId, qc, savedContent, savedId, savedIsRunning, savedStreamSeq]);

  const generate = useCallback(async () => {
    if (runningRef.current) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    runningRef.current = true;
    bufferRef.current = "";
    setStreamContent("");
    setIsStreaming(true);
    setError(null);

    const flush = () => {
      rafRef.current = null;
      setStreamContent(bufferRef.current);
    };

    try {
      for await (const token of readSSE(
        "/algorithms/generate",
        { document_id: documentId, mode: activeMode },
        controller.signal,
      )) {
        bufferRef.current += token;
        if (rafRef.current === null) {
          rafRef.current = requestAnimationFrame(flush);
        }
      }
    } catch (e) {
      if (controller.signal.aborted) return;
      setError(e instanceof Error ? e.message : "Ошибка генерации");
    } finally {
      runningRef.current = false;
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      if (!controller.signal.aborted) {
        setStreamContent(bufferRef.current);
        setIsStreaming(false);
        qc.invalidateQueries({ queryKey: ["algorithm-by-doc", documentId] });
      }
    }
  }, [activeMode, documentId, qc]);

  return {
    content,
    isStreaming: isStreaming || savedIsRunning,
    isLoadingSaved,
    error,
    generate,
    hasSaved: saved?.status === "completed",
    isRunningSaved: savedIsRunning,
    activeMode,
  };
}

export function algorithmPdfFilename(documentName: string): string {
  const diagnosis = documentName
    .normalize("NFKC")
    .replace(/\.pdf$/i, "")
    .replace(
      /^(?:Клинический\s+)?Алгоритм(?:\s+оказания\s+медицинской\s+помощи)?\s*:\s*/i,
      "",
    );
  const withoutControlCharacters = Array.from(diagnosis, (character) =>
    (character.codePointAt(0) ?? 0) < 32 ? " " : character,
  ).join("");
  const safeName = withoutControlCharacters
    .replace(/[<>:"/\\|?*]/g, " ")
    .replace(/\s+/g, "_")
    .replace(/_+/g, "_")
    .replace(/^[._]+|[._]+$/g, "");
  const shortenedName = Array.from(safeName || "клинический").slice(0, 100).join("");

  return `Алгоритм_${shortenedName.replace(/[._]+$/g, "")}.pdf`;
}

export async function exportPdf(markdown: string, documentName: string): Promise<void> {
  const filename = algorithmPdfFilename(documentName);
  const blob = await apiBlob("/algorithms/export-pdf", {
    method: "POST",
    body: JSON.stringify({ markdown }),
  });
  if (blob.size === 0) {
    throw new Error("Сервер вернул пустой PDF");
  }

  const pdfFile = new File([blob], filename, { type: blob.type || "application/pdf" });
  const url = URL.createObjectURL(pdfFile);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.style.display = "none";
  document.body.appendChild(link);
  link.click();

  // Keep the blob alive until the browser has accepted the download.
  window.setTimeout(() => {
    link.remove();
    URL.revokeObjectURL(url);
  }, 1000);
}
