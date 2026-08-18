import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";

interface AppConfig {
  upload_max_mb: number;
}

const FALLBACK_UPLOAD_MAX_MB = 50;

export function useAppConfig() {
  return useQuery<AppConfig>({
    queryKey: ["app-config"],
    queryFn: () => apiFetch("/config"),
    staleTime: Infinity,
    retry: 1,
  });
}

export function useUploadMaxBytes(): number {
  const { data } = useAppConfig();
  return (data?.upload_max_mb ?? FALLBACK_UPLOAD_MAX_MB) * 1024 * 1024;
}
