import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";

export interface CurrentGenerationStats {
  running_algorithms: number;
  running_chats: number;
  active_sessions: number;
  processing_documents: number;
}

export interface GpuMetric {
  index: number;
  name: string;
  utilization_gpu_percent: number | null;
  memory_used_mb: number | null;
  memory_total_mb: number | null;
  temperature_c: number | null;
  power_draw_w: number | null;
}

export interface SystemSnapshot {
  load_1m: number | null;
  load_5m: number | null;
  load_15m: number | null;
  cpu_count: number;
  memory_total_kb: number | null;
  memory_available_kb: number | null;
  memory_used_percent: number | null;
  gpu_metrics: GpuMetric[];
}

export interface ActionCount {
  action: string;
  count: number;
}

export interface UserActivitySummary {
  user_id: string | null;
  username: string;
  user_status: "active" | "deleted" | "system";
  algorithm_generations: number;
  chat_generations: number;
  admin_actions: number;
}

export interface SystemMetricSample extends SystemSnapshot {
  id: number;
  active_sessions: number;
  running_algorithms: number;
  running_chats: number;
  processing_documents: number;
  created_at: string;
}

export interface AuditLogEntry {
  id: number;
  actor_user_id: string | null;
  actor_username: string | null;
  target_user_id: string | null;
  target_username: string | null;
  action: string;
  entity_type: string;
  entity_id: string | null;
  payload: string;
  created_at: string;
}

export interface MonitoringSummary {
  scope: "all" | "managed";
  from: string;
  to: string;
  current: CurrentGenerationStats;
  system: SystemSnapshot | null;
  history: ActionCount[];
  top_users: UserActivitySummary[];
  metrics: SystemMetricSample[];
  logs: AuditLogEntry[];
}

export function useMonitoring(from: string, to: string) {
  return useQuery<MonitoringSummary>({
    queryKey: ["monitoring", from, to],
    queryFn: () =>
      apiFetch(`/monitoring?from=${encodeURIComponent(from)}&to=${encodeURIComponent(to)}`),
    refetchInterval: 5000,
  });
}
