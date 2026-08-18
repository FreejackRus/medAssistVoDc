import { parseSseStream } from "@/lib/sseParser";
import { HttpError } from "@/lib/httpError";

const BASE_URL = "/api";
let unauthorizedHandler: (() => void) | null = null;

export function setUnauthorizedHandler(handler: (() => void) | null): void {
  unauthorizedHandler = handler;
}

function humanError(status: number, body: string): string {
  // Try to extract message from JSON body like {"error":"..."} or {"detail":"..."}
  try {
    const json = JSON.parse(body);
    const msg = json.error || json.detail || json.message;
    if (typeof msg === "string" && msg.length > 0) return msg;
  } catch { /* not JSON */ }

  switch (status) {
    case 400: return body || "Некорректный запрос";
    case 401: return body || "Требуется вход в систему";
    case 403: return body || "Недостаточно прав";
    case 404: return "Ресурс не найден";
    case 409: return body || "Конфликт данных";
    case 413: return "Файл слишком большой";
    case 422: return body || "Ошибка валидации";
    case 500: return "Внутренняя ошибка сервера. Попробуйте позже";
    case 502: case 503: case 504: return "Сервер временно недоступен. Попробуйте позже";
    default: return body || `Ошибка (${status})`;
  }
}

function networkError(e: unknown): Error {
  if (e instanceof TypeError && (e.message.includes("fetch") || e.message.includes("network"))) {
    return new Error("Нет соединения с сервером. Проверьте, что все сервисы запущены");
  }
  return e instanceof Error ? e : new Error("Неизвестная ошибка");
}

async function responseError(res: Response): Promise<HttpError> {
  const body = await res.text().catch(() => "");
  const error = new HttpError(res.status, body, humanError(res.status, body));
  if (res.status === 401) unauthorizedHandler?.();
  return error;
}

export async function apiFetch<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  let res: Response;
  try {
    const headers = new Headers(options?.headers);
    if (options?.body) {
      if (!headers.has("Content-Type")) headers.set("Content-Type", "application/json");
    }
    res = await fetch(`${BASE_URL}${path}`, {
      ...options,
      headers,
      credentials: "include",
    });
  } catch (e) {
    throw networkError(e);
  }
  if (!res.ok) {
    throw await responseError(res);
  }
  return res.json();
}

export async function apiBlob(
  path: string,
  options?: RequestInit,
): Promise<Blob> {
  let res: Response;
  try {
    const headers = new Headers(options?.headers);
    if (options?.body && !(options.body instanceof FormData)) {
      if (!headers.has("Content-Type")) headers.set("Content-Type", "application/json");
    }
    res = await fetch(`${BASE_URL}${path}`, {
      ...options,
      headers,
      credentials: "include",
    });
  } catch (e) {
    throw networkError(e);
  }
  if (!res.ok) {
    throw await responseError(res);
  }
  return res.blob();
}

export async function apiUpload<T>(
  path: string,
  formData: FormData,
  signal?: AbortSignal,
): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE_URL}${path}`, {
      method: "POST",
      body: formData,
      signal,
      credentials: "include",
    });
  } catch (e) {
    throw networkError(e);
  }
  if (!res.ok) {
    throw await responseError(res);
  }
  return res.json();
}

export async function* readSSE(
  path: string,
  body: unknown,
  signal?: AbortSignal,
): AsyncGenerator<string> {
  yield* readTokenStream(
    path,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    },
  );
}

export async function* readUploadSSE(
  path: string,
  formData: FormData,
  signal?: AbortSignal,
): AsyncGenerator<string> {
  yield* readTokenStream(
    path,
    {
      method: "POST",
      body: formData,
      signal,
    },
  );
}

async function openStream(path: string, options: RequestInit): Promise<ReadableStream<Uint8Array>> {
  let res: Response;
  try {
    res = await fetch(`${BASE_URL}${path}`, {
      ...options,
      credentials: "include",
    });
  } catch (e) {
    throw networkError(e);
  }
  if (!res.ok || !res.body) {
    throw await responseError(res);
  }
  return res.body;
}

async function* readTokenStream(
  path: string,
  options: RequestInit,
): AsyncGenerator<string> {
  const stream = await openStream(path, options);
  let deferredError: Error | null = null;

  for await (const frame of parseSseStream(stream, "message")) {
    if (frame.event === "save_error" || frame.event === "error") {
      let message =
        frame.event === "save_error"
          ? "Не удалось сохранить результат"
          : "Генерация завершилась ошибкой";
      try {
        const parsed = JSON.parse(frame.data);
        if (typeof parsed === "string") message = parsed;
      } catch {
        // Keep the localized fallback.
      }
      deferredError = new Error(message);
      continue;
    }

    try {
      const parsed: unknown = JSON.parse(frame.data);
      if (typeof parsed === "string") yield parsed;
    } catch {
      // Skip malformed SSE data.
    }
  }

  if (deferredError) throw deferredError;
}

export interface ResumeStreamEvent {
  seq: number;
  event: "token" | "done" | "error";
  content: string;
}

export async function* readResumeSSE(
  path: string,
  signal?: AbortSignal,
): AsyncGenerator<ResumeStreamEvent> {
  const stream = await openStream(path, {
    signal,
  });
  for await (const frame of parseSseStream(stream, "token")) {
    const event = frame.event === "done" || frame.event === "error" ? frame.event : "token";
    let content = "";
    try {
      const parsed = JSON.parse(frame.data);
      if (typeof parsed === "string") content = parsed;
    } catch {
      content = frame.data;
    }
    yield {
      seq: frame.id,
      event,
      content,
    };
  }
}

export interface AccountEvent {
  id: number;
  event: string;
  payload: Record<string, unknown>;
}

export async function* readAccountEvents(
  after = 0,
  signal?: AbortSignal,
): AsyncGenerator<AccountEvent> {
  const stream = await openStream(`/events?after=${after}`, {
    signal,
  });
  for await (const frame of parseSseStream(stream, "message")) {
    let payload: Record<string, unknown> = {};
    try {
      const parsed = JSON.parse(frame.data);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        payload = parsed as Record<string, unknown>;
      }
    } catch {
      payload = {};
    }
    yield { id: frame.id, event: frame.event, payload };
  }
}
