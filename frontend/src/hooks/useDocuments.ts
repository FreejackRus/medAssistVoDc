import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiFetch, apiUpload } from "@/lib/api";

export interface Document {
  id: string;
  user_id: string;
  filename: string;
  diagnosis_name: string | null;
  mkb_code: string | null;
  status: "processing" | "ready" | "error";
  chunk_count: number;
  created_at: string;
  updated_at: string;
}

export function useDocuments() {
  return useQuery<Document[]>({
    queryKey: ["documents"],
    queryFn: () => apiFetch("/documents"),
    refetchInterval: (query) => {
      const docs = query.state.data;
      if (docs?.some((d) => d.status === "processing")) return 3000;
      return false;
    },
  });
}

export function useUploadDocument() {
  const qc = useQueryClient();
  return useMutation<Document, Error, { file: File; signal?: AbortSignal }>({
    mutationFn: ({ file, signal }) => {
      const fd = new FormData();
      fd.append("pdf", file);
      return apiUpload("/documents/upload", fd, signal);
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["documents"] }),
  });
}

export function useDeleteDocument() {
  const qc = useQueryClient();
  return useMutation<unknown, Error, string>({
    mutationFn: (id) =>
      apiFetch(`/documents/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["documents"] }),
  });
}

export function useRetryDocument() {
  const qc = useQueryClient();
  return useMutation<Document, Error, string>({
    mutationFn: (id) =>
      apiFetch(`/documents/${id}/retry`, { method: "POST" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["documents"] }),
  });
}

export function useImportRecommendation() {
  const qc = useQueryClient();
  return useMutation<Document, Error, { code_version: string; title: string }>({
    mutationFn: (body) =>
      apiFetch("/documents/import-recommendation", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["documents"] }),
  });
}
