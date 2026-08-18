import { useMemo, useState } from "react";
import type { ComponentType } from "react";
import { Activity, Cpu, DatabaseZap, FileClock, Gauge, RefreshCw, ShieldCheck } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useAuth } from "@/hooks/useAuth";
import {
  useMonitoring,
  type AuditLogEntry,
  type CurrentGenerationStats,
  type GpuMetric,
  type SystemMetricSample,
  type UserActivitySummary,
} from "@/hooks/useMonitoring";

const actionLabels: Record<string, string> = {
  user_created: "Пользователь создан",
  user_updated: "Карточка пользователя изменена",
  user_deleted: "Пользователь удален",
  user_password_reset: "Пароль сброшен",
  algorithm_generation_started: "Алгоритм запущен",
  algorithm_generation_completed: "Алгоритм завершен",
  algorithm_generation_error: "Ошибка алгоритма",
  chat_generation_started: "Ответ в чате запущен",
  chat_generation_completed: "Ответ в чате завершен",
  chat_generation_error: "Ошибка ответа в чате",
};

function toInputDate(value: Date) {
  const offset = value.getTimezoneOffset() * 60_000;
  return new Date(value.getTime() - offset).toISOString().slice(0, 16);
}

type MonitoringTab = "overview" | "load" | "activity" | "logs";

export default function MonitoringPage() {
  const { user } = useAuth();
  const isAdmin = user?.role === "admin";
  const [activeTab, setActiveTab] = useState<MonitoringTab>("overview");
  const [from, setFrom] = useState(() => toInputDate(new Date(Date.now() - 24 * 60 * 60 * 1000)));
  const [to, setTo] = useState(() => toInputDate(new Date()));
  const { data, isLoading, refetch, isFetching } = useMonitoring(from, to);

  const history = useMemo(() => {
    const map = new Map((data?.history ?? []).map((item) => [item.action, item.count]));
    return {
      algorithmStarted: map.get("algorithm_generation_started") ?? 0,
      algorithmCompleted: map.get("algorithm_generation_completed") ?? 0,
      algorithmError: map.get("algorithm_generation_error") ?? 0,
      chatStarted: map.get("chat_generation_started") ?? 0,
      chatCompleted: map.get("chat_generation_completed") ?? 0,
      chatError: map.get("chat_generation_error") ?? 0,
      userCreated: map.get("user_created") ?? 0,
      userUpdated: map.get("user_updated") ?? 0,
      userDeleted: map.get("user_deleted") ?? 0,
      passwordReset: map.get("user_password_reset") ?? 0,
    };
  }, [data?.history]);

  const tabs: Array<{ id: MonitoringTab; label: string }> =
    isAdmin
      ? [
          { id: "overview", label: "Обзор" },
          { id: "load", label: "Нагрузка" },
          { id: "activity", label: "Активность" },
          { id: "logs", label: "Журнал" },
        ]
      : [
          { id: "overview", label: "Обзор" },
          { id: "activity", label: "Активность" },
          { id: "logs", label: "Журнал" },
        ];
  const visibleTab = tabs.some((tab) => tab.id === activeTab) ? activeTab : "overview";

  return (
    <div className="flex h-full min-h-0 flex-col gap-6 overflow-hidden p-6">
      <div className="shrink-0 flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <h2 className="text-2xl font-bold">Мониторинг</h2>
          <p className="text-sm text-muted-foreground">
            {data?.scope === "managed"
              ? "Показаны ваши действия, ваши пользователи и текущие генерации в вашей зоне."
              : "Текущие генерации, нагрузка сервера и журнал действий по системе."}
          </p>
        </div>
        <div className="grid gap-2 sm:grid-cols-[1fr_1fr_auto]">
          <Input type="datetime-local" value={from} onChange={(e) => setFrom(e.target.value)} />
          <Input type="datetime-local" value={to} onChange={(e) => setTo(e.target.value)} />
          <Button variant="outline" className="gap-2" onClick={() => void refetch()}>
            <RefreshCw className={isFetching ? "h-4 w-4 animate-spin" : "h-4 w-4"} />
            Обновить
          </Button>
        </div>
      </div>

      <div className="flex shrink-0 flex-wrap gap-2 border-b pb-2">
        {tabs.map((tab) => (
          <Button
            key={tab.id}
            variant={visibleTab === tab.id ? "default" : "ghost"}
            size="sm"
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </Button>
        ))}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto pr-1">
        {visibleTab === "overview" && (
          <div className="space-y-3">
            <CurrentStatsCards current={data?.current} />
            <div className="grid gap-3 xl:grid-cols-[minmax(0,1.2fr)_minmax(360px,0.8fr)] xl:items-start">
              {isAdmin && data?.system ? (
                <SystemLoadCard system={data.system} />
              ) : (
                <HistorySummaryCard history={history} />
              )}
              <div className="grid gap-3">
                {isAdmin && data?.system && <HistorySummaryCard history={history} />}
                {isAdmin && <UserActivityCard users={data?.top_users ?? []} />}
              </div>
            </div>
          </div>
        )}

        {visibleTab === "load" && isAdmin && (
          <div className="grid gap-3 xl:grid-cols-[minmax(360px,0.75fr)_minmax(0,1.25fr)] xl:items-start">
            {data?.system && <SystemLoadCard system={data.system} />}
            <LoadHistoryCard metrics={data?.metrics ?? []} isLoading={isLoading} />
          </div>
        )}

        {visibleTab === "activity" && (
          <div className="space-y-3">
            {isAdmin && <CurrentStatsCards current={data?.current} />}
            {isAdmin ? (
              <div className="grid gap-3 xl:grid-cols-2 xl:items-start">
                <HistorySummaryCard history={history} />
                <UserActivityCard users={data?.top_users ?? []} />
              </div>
            ) : (
              <UserActivityCard users={data?.top_users ?? []} />
            )}
          </div>
        )}

        {visibleTab === "logs" && <LogsCard logs={data?.logs ?? []} />}
      </div>
    </div>
  );
}

function CurrentStatsCards({ current }: { current?: CurrentGenerationStats }) {
  return (
    <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
      <MetricCard
        icon={Activity}
        label="Алгоритмы пишутся"
        value={current?.running_algorithms ?? 0}
      />
      <MetricCard icon={DatabaseZap} label="Ответы в чатах" value={current?.running_chats ?? 0} />
      <MetricCard icon={ShieldCheck} label="Активные входы" value={current?.active_sessions ?? 0} />
      <MetricCard
        icon={FileClock}
        label="Документы обрабатываются"
        value={current?.processing_documents ?? 0}
      />
    </div>
  );
}

function SystemLoadCard({
  system,
}: {
  system: {
    load_1m: number | null;
    load_5m: number | null;
    load_15m: number | null;
    cpu_count: number;
    memory_available_kb: number | null;
    memory_used_percent: number | null;
    gpu_metrics: GpuMetric[];
  };
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Cpu className="h-4 w-4" />
          Нагрузка сервера
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <InfoStat label="Load 1/5/15" value={formatLoad(system)} />
          <InfoStat label="CPU" value={`${system.cpu_count || "—"} ядер`} />
          <InfoStat label="Память" value={formatMemory(system.memory_used_percent)} />
          <InfoStat label="Доступно RAM" value={formatKb(system.memory_available_kb)} />
        </div>
        <div className="mt-4">
          <div className="mb-2 flex items-center gap-2 text-sm font-medium">
            <Gauge className="h-4 w-4 text-muted-foreground" />
            GPU
          </div>
          {system.gpu_metrics.length === 0 ? (
            <p className="text-sm text-muted-foreground">GPU-метрики недоступны для backend-контейнера.</p>
          ) : (
            <div className="grid gap-3 xl:grid-cols-2">
              {system.gpu_metrics.map((gpu) => (
                <div key={gpu.index} className="rounded-md bg-muted/40 p-3">
                  <div className="mb-2 flex items-center justify-between gap-3">
                    <div className="min-w-0">
                      <p className="font-medium">GPU {gpu.index}</p>
                      <p className="truncate text-xs text-muted-foreground">{gpu.name}</p>
                    </div>
                    <Badge variant="secondary">{formatPercent(gpu.utilization_gpu_percent)}</Badge>
                  </div>
                  <div className="grid gap-2 sm:grid-cols-2">
                    <InfoStat className="sm:col-span-2" label="VRAM" value={formatGpuMemory(gpu)} />
                    <InfoStat label="Температура" value={formatTemperature(gpu.temperature_c)} />
                    <InfoStat label="Питание" value={formatPower(gpu.power_draw_w)} />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function HistorySummaryCard({
  history,
}: {
  history: {
    algorithmStarted: number;
    algorithmCompleted: number;
    algorithmError: number;
    chatStarted: number;
    chatCompleted: number;
    chatError: number;
    userCreated: number;
    userUpdated: number;
    userDeleted: number;
    passwordReset: number;
  };
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">История за период</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid gap-2 sm:grid-cols-2">
          <InfoStat label="Запусков алгоритмов" value={history.algorithmStarted} />
          <InfoStat label="Завершено алгоритмов" value={history.algorithmCompleted} />
          <InfoStat label="Запусков чата" value={history.chatStarted} />
          <InfoStat label="Ответов завершено" value={history.chatCompleted} />
          <InfoStat label="Ошибок генерации" value={history.algorithmError + history.chatError} />
          <InfoStat
            label="Админ-действий"
            value={history.userCreated + history.userUpdated + history.userDeleted + history.passwordReset}
          />
        </div>
      </CardContent>
    </Card>
  );
}

function UserActivityCard({ users }: { users: UserActivitySummary[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Активность пользователей</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {users.length === 0 ? (
            <p className="text-sm text-muted-foreground">За выбранный период действий нет.</p>
          ) : (
            users.map((item) => (
              <div
                key={`${item.user_id ?? item.username}`}
                className="grid gap-2 rounded-md border p-2 text-sm sm:grid-cols-[1fr_auto]"
              >
                <span className="flex min-w-0 flex-wrap items-center gap-2 font-medium">
                  <span>{item.username}</span>
                  {item.user_status === "deleted" && <Badge variant="outline">удален</Badge>}
                  {item.user_status === "system" && <Badge variant="secondary">система</Badge>}
                </span>
                <span className="text-muted-foreground">
                  алгоритмы: {item.algorithm_generations} · чат: {item.chat_generations} · учетные записи: {item.admin_actions}
                </span>
              </div>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function LoadHistoryCard({
  metrics,
  isLoading,
}: {
  metrics: SystemMetricSample[];
  isLoading: boolean;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">История нагрузки</CardTitle>
      </CardHeader>
      <CardContent>
        <LoadChart metrics={metrics} />
        <div className="max-h-[24rem] overflow-auto">
          <table className="w-full min-w-[900px] text-left text-sm">
            <thead className="sticky top-0 border-b bg-card text-xs text-muted-foreground">
              <tr>
                <th className="py-2 pr-3 font-medium">Время</th>
                <th className="py-2 pr-3 font-medium">Load</th>
                <th className="py-2 pr-3 font-medium">RAM</th>
                <th className="py-2 pr-3 font-medium">GPU</th>
                <th className="py-2 pr-3 font-medium">Алгоритмы</th>
                <th className="py-2 pr-3 font-medium">Чаты</th>
                <th className="py-2 pr-3 font-medium">Сессии</th>
                <th className="py-2 pr-3 font-medium">Документы</th>
              </tr>
            </thead>
            <tbody>
              {metrics.slice(0, 20).map((item) => (
                <tr key={item.id} className="border-b last:border-0">
                  <td className="py-2 pr-3">{item.created_at}</td>
                  <td className="py-2 pr-3">{formatLoad(item)}</td>
                  <td className="py-2 pr-3">{formatMemory(item.memory_used_percent)}</td>
                  <td className="py-2 pr-3">{formatGpuSummary(item.gpu_metrics)}</td>
                  <td className="py-2 pr-3">{item.running_algorithms}</td>
                  <td className="py-2 pr-3">{item.running_chats}</td>
                  <td className="py-2 pr-3">{item.active_sessions}</td>
                  <td className="py-2 pr-3">{item.processing_documents}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {!isLoading && metrics.length === 0 && (
            <p className="py-4 text-sm text-muted-foreground">Сэмплов за период нет.</p>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function LogsCard({ logs }: { logs: AuditLogEntry[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Журнал действий</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="max-h-[calc(100vh-18rem)] min-h-[20rem] space-y-2 overflow-y-auto pr-1">
          {logs.length === 0 ? (
            <p className="text-sm text-muted-foreground">За выбранный период записей нет.</p>
          ) : (
            logs.map((item) => <LogRow key={item.id} item={item} />)
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function MetricCard({
  icon: Icon,
  label,
  value,
}: {
  icon: ComponentType<{ className?: string }>;
  label: string;
  value: number;
}) {
  return (
    <Card>
      <CardContent className="flex items-center justify-between gap-3">
        <div>
          <p className="text-sm text-muted-foreground">{label}</p>
          <p className="text-3xl font-semibold">{value}</p>
        </div>
        <Icon className="h-6 w-6 text-muted-foreground" />
      </CardContent>
    </Card>
  );
}

function InfoStat({
  label,
  value,
  className = "",
}: {
  label: string;
  value: string | number;
  className?: string;
}) {
  return (
    <div className={`rounded-md bg-muted/50 px-3 py-2 ${className}`}>
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-base font-semibold">{value}</p>
    </div>
  );
}

function LogRow({ item }: { item: AuditLogEntry }) {
  const details = parsePayload(item.payload);
  return (
    <div className="grid gap-2 rounded-md border p-3 text-sm lg:grid-cols-[180px_1fr_auto]">
      <div className="text-muted-foreground">{item.created_at}</div>
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={item.action.includes("error") || item.action.includes("deleted") ? "destructive" : "secondary"}>
            {actionLabels[item.action] ?? item.action}
          </Badge>
          <span className="font-medium">{item.actor_username ?? "system"}</span>
          {item.target_username && <span className="text-muted-foreground">→ {item.target_username}</span>}
        </div>
        {details && <p className="mt-1 break-words text-xs text-muted-foreground">{details}</p>}
      </div>
      <div className="text-xs text-muted-foreground">{item.entity_type}</div>
    </div>
  );
}

function parsePayload(raw: string) {
  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const parts: string[] = [];
    if (Array.isArray(parsed.changed_fields) && parsed.changed_fields.length > 0) {
      parts.push(`поля: ${parsed.changed_fields.join(", ")}`);
    }
    if (typeof parsed.document_id === "string") parts.push(`document_id: ${parsed.document_id}`);
    if (typeof parsed.session_id === "string") parts.push(`session_id: ${parsed.session_id}`);
    if (typeof parsed.role === "string") parts.push(`роль: ${parsed.role}`);
    if (typeof parsed.tokens === "number") parts.push(`токены: ${parsed.tokens}`);
    return parts.join(" · ");
  } catch {
    return "";
  }
}

function formatLoad(value: { load_1m: number | null; load_5m: number | null; load_15m: number | null }) {
  return [value.load_1m, value.load_5m, value.load_15m]
    .map((v) => (typeof v === "number" ? v.toFixed(2) : "—"))
    .join(" / ");
}

function formatMemory(value: number | null) {
  return typeof value === "number" ? `${value.toFixed(1)}%` : "—";
}

function formatKb(value: number | null) {
  if (typeof value !== "number") return "—";
  return `${(value / 1024 / 1024).toFixed(1)} ГБ`;
}

function formatPercent(value: number | null) {
  return typeof value === "number" ? `${value.toFixed(0)}%` : "—";
}

function formatTemperature(value: number | null) {
  return typeof value === "number" ? `${value.toFixed(0)} °C` : "—";
}

function formatPower(value: number | null) {
  return typeof value === "number" ? `${value.toFixed(1)} Вт` : "—";
}

function formatGpuMemory(gpu: GpuMetric) {
  if (typeof gpu.memory_used_mb !== "number" || typeof gpu.memory_total_mb !== "number") {
    return "—";
  }
  return `${(gpu.memory_used_mb / 1024).toFixed(1)} / ${(gpu.memory_total_mb / 1024).toFixed(1)} ГБ`;
}

function formatGpuSummary(gpus: GpuMetric[]) {
  if (gpus.length === 0) return "—";
  return gpus
    .map((gpu) => `#${gpu.index}: ${formatPercent(gpu.utilization_gpu_percent)}, ${formatGpuMemory(gpu)}`)
    .join(" · ");
}

function LoadChart({ metrics }: { metrics: SystemMetricSample[] }) {
  const points = useMemo(() => {
    return metrics
      .slice()
      .reverse()
      .map((item) => ({
        id: item.id,
        label: formatTimeLabel(item.created_at),
        ram: item.memory_used_percent ?? null,
        gpu: maxGpuUsage(item.gpu_metrics),
      }));
  }, [metrics]);

  if (points.length < 2) {
    return (
      <div className="mb-4 flex h-48 items-center justify-center rounded-md border bg-muted/20 text-sm text-muted-foreground">
        Для графика нужно минимум два сэмпла за выбранный период.
      </div>
    );
  }

  const width = 720;
  const height = 220;
  const padding = { top: 18, right: 20, bottom: 30, left: 36 };
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;
  const xFor = (index: number) => padding.left + (index / Math.max(points.length - 1, 1)) * innerWidth;
  const yFor = (value: number) => padding.top + (1 - value / 100) * innerHeight;
  const ramPath = buildChartPath(points.map((point) => point.ram), xFor, yFor);
  const gpuPath = buildChartPath(points.map((point) => point.gpu), xFor, yFor);
  const first = points[0]?.label ?? "";
  const last = points[points.length - 1]?.label ?? "";

  return (
    <div className="mb-4 rounded-md border bg-background p-3">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2 text-sm">
        <div className="font-medium">RAM и максимальная загрузка GPU</div>
        <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
          <Legend colorClass="bg-emerald-500" label="RAM" />
          <Legend colorClass="bg-sky-500" label="GPU max" />
        </div>
      </div>
      <svg className="h-56 w-full overflow-visible" viewBox={`0 0 ${width} ${height}`} role="img">
        {[0, 50, 100].map((value) => (
          <g key={value}>
            <line
              x1={padding.left}
              x2={width - padding.right}
              y1={yFor(value)}
              y2={yFor(value)}
              className="stroke-border"
              strokeDasharray={value === 0 ? undefined : "4 4"}
            />
            <text x={8} y={yFor(value) + 4} className="fill-muted-foreground text-[10px]">
              {value}%
            </text>
          </g>
        ))}
        {ramPath && <path d={ramPath} fill="none" stroke="rgb(16 185 129)" strokeWidth="2.5" />}
        {gpuPath && <path d={gpuPath} fill="none" stroke="rgb(14 165 233)" strokeWidth="2.5" />}
        <text x={padding.left} y={height - 8} className="fill-muted-foreground text-[10px]">
          {first}
        </text>
        <text
          x={width - padding.right}
          y={height - 8}
          textAnchor="end"
          className="fill-muted-foreground text-[10px]"
        >
          {last}
        </text>
      </svg>
    </div>
  );
}

function Legend({ colorClass, label }: { colorClass: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className={`h-2.5 w-2.5 rounded-full ${colorClass}`} />
      {label}
    </span>
  );
}

function buildChartPath(
  values: Array<number | null>,
  xFor: (index: number) => number,
  yFor: (value: number) => number,
) {
  let path = "";
  values.forEach((value, index) => {
    if (typeof value !== "number") {
      return;
    }
    path += `${path ? " L" : "M"} ${xFor(index).toFixed(1)} ${yFor(value).toFixed(1)}`;
  });
  return path;
}

function maxGpuUsage(gpus: GpuMetric[]) {
  const values = gpus
    .map((gpu) => gpu.utilization_gpu_percent)
    .filter((value): value is number => typeof value === "number");
  if (values.length === 0) return null;
  return Math.max(...values);
}

function formatTimeLabel(value: string) {
  return value.slice(11, 16) || value;
}
