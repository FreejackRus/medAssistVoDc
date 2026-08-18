const BASE_URL = "/api";
const TOKEN_KEY = "clinical_ai_token";

export function getAuthToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setAuthToken(token: string | null) {
  if (token) {
    localStorage.setItem(TOKEN_KEY, token);
  } else {
    localStorage.removeItem(TOKEN_KEY);
  }
}

function authHeaders(headers?: HeadersInit): HeadersInit {
  const next: Record<string, string> = { ...(headers as Record<string, string> | undefined) };
  const token = getAuthToken();
  if (token) next.Authorization = `Bearer ${token}`;
  return next;
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

export async function apiFetch<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  let res: Response;
  try {
    const headers: HeadersInit = authHeaders(options?.headers);
    if (options?.body) {
      (headers as Record<string, string>)["Content-Type"] ??= "application/json";
    }
    res = await fetch(`${BASE_URL}${path}`, {
      ...options,
      headers,
    });
  } catch (e) {
    throw networkError(e);
  }
  if (!res.ok) {
    const body = await res.text();
    throw new Error(humanError(res.status, body));
  }
  return res.json();
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
      headers: authHeaders(),
      signal,
    });
  } catch (e) {
    throw networkError(e);
  }
  if (!res.ok) {
    const body = await res.text();
    throw new Error(humanError(res.status, body));
  }
  return res.json();
}

export async function* readSSE(
  path: string,
  body: unknown,
  signal?: AbortSignal,
): AsyncGenerator<string> {
  let res: Response;
  try {
    res = await fetch(`${BASE_URL}${path}`, {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(body),
      signal,
    });
  } catch (e) {
    throw networkError(e);
  }
  if (!res.ok || !res.body) {
    const errBody = await res.text().catch(() => "");
    throw new Error(humanError(res.status, errBody));
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let currentEvent = "message";
  let deferredError: Error | null = null;

  const handleLine = function* (line: string): Generator<string> {
    if (line === "") {
      currentEvent = "message";
      return;
    }
    if (line.startsWith("event: ")) {
      currentEvent = line.slice(7).trim();
      return;
    }
    if (!line.startsWith("data: ")) return;
    const raw = line.slice(6);
    if (currentEvent === "save_error" || currentEvent === "error") {
      let message =
        currentEvent === "save_error"
          ? "Не удалось сохранить результат"
          : "Генерация завершилась ошибкой";
      try {
        const parsed = JSON.parse(raw);
        if (typeof parsed === "string") message = parsed;
      } catch {
        // keep default message
      }
      deferredError = new Error(message);
      return;
    }
    try {
      yield JSON.parse(raw);
    } catch {
      // skip malformed SSE data
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      yield* handleLine(line);
    }
  }

  if (buffer.length > 0) {
    yield* handleLine(buffer);
  }

  if (deferredError) throw deferredError;
}

export async function* readUploadSSE(
  path: string,
  formData: FormData,
  signal?: AbortSignal,
): AsyncGenerator<string> {
  let res: Response;
  try {
    res = await fetch(`${BASE_URL}${path}`, {
      method: "POST",
      headers: authHeaders(),
      body: formData,
      signal,
    });
  } catch (e) {
    throw networkError(e);
  }
  if (!res.ok || !res.body) {
    const errBody = await res.text().catch(() => "");
    throw new Error(humanError(res.status, errBody));
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let currentEvent = "message";
  let deferredError: Error | null = null;

  const handleLine = function* (line: string): Generator<string> {
    if (line === "") {
      currentEvent = "message";
      return;
    }
    if (line.startsWith("event: ")) {
      currentEvent = line.slice(7).trim();
      return;
    }
    if (!line.startsWith("data: ")) return;
    const raw = line.slice(6);
    if (currentEvent === "save_error" || currentEvent === "error") {
      let message =
        currentEvent === "save_error"
          ? "Не удалось сохранить результат"
          : "Генерация завершилась ошибкой";
      try {
        const parsed = JSON.parse(raw);
        if (typeof parsed === "string") message = parsed;
      } catch {
        // keep default message
      }
      deferredError = new Error(message);
      return;
    }
    try {
      yield JSON.parse(raw);
    } catch {
      // skip malformed SSE data
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      yield* handleLine(line);
    }
  }

  if (buffer.length > 0) {
    yield* handleLine(buffer);
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
  let res: Response;
  try {
    res = await fetch(`${BASE_URL}${path}`, {
      headers: authHeaders(),
      signal,
    });
  } catch (e) {
    throw networkError(e);
  }
  if (!res.ok || !res.body) {
    const errBody = await res.text().catch(() => "");
    throw new Error(humanError(res.status, errBody));
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let currentEvent = "token";
  let currentId = 0;
  let dataLines: string[] = [];

  const dispatch = (): ResumeStreamEvent | null => {
    if (dataLines.length === 0) {
      currentEvent = "token";
      currentId = 0;
      return null;
    }

    const event =
      currentEvent === "done" || currentEvent === "error" ? currentEvent : "token";
    const raw = dataLines.join("\n");
    let content = "";
    try {
      const parsed = JSON.parse(raw);
      if (typeof parsed === "string") content = parsed;
    } catch {
      content = raw;
    }

    const result: ResumeStreamEvent = {
      seq: currentId,
      event,
      content,
    };
    currentEvent = "token";
    currentId = 0;
    dataLines = [];
    return result;
  };

  const handleLine = (line: string): ResumeStreamEvent | null => {
    if (line === "") return dispatch();
    if (line.startsWith("id: ")) {
      const parsed = Number.parseInt(line.slice(4).trim(), 10);
      currentId = Number.isFinite(parsed) ? parsed : 0;
      return null;
    }
    if (line.startsWith("event: ")) {
      currentEvent = line.slice(7).trim();
      return null;
    }
    if (line.startsWith("data: ")) {
      dataLines.push(line.slice(6));
    }
    return null;
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const rawLine of lines) {
      const event = handleLine(rawLine.replace(/\r$/, ""));
      if (event) yield event;
    }
  }

  if (buffer.length > 0) {
    const event = handleLine(buffer.replace(/\r$/, ""));
    if (event) yield event;
  }

  const event = dispatch();
  if (event) yield event;
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
  let res: Response;
  try {
    res = await fetch(`${BASE_URL}/events?after=${after}`, {
      headers: authHeaders(),
      signal,
    });
  } catch (e) {
    throw networkError(e);
  }
  if (!res.ok || !res.body) {
    const errBody = await res.text().catch(() => "");
    throw new Error(humanError(res.status, errBody));
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let currentEvent = "message";
  let currentId = 0;
  let dataLines: string[] = [];

  const dispatch = (): AccountEvent | null => {
    if (dataLines.length === 0) {
      currentEvent = "message";
      currentId = 0;
      return null;
    }
    const raw = dataLines.join("\n");
    let payload: Record<string, unknown> = {};
    try {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        payload = parsed as Record<string, unknown>;
      }
    } catch {
      payload = {};
    }
    const result = { id: currentId, event: currentEvent, payload };
    currentEvent = "message";
    currentId = 0;
    dataLines = [];
    return result;
  };

  const handleLine = (line: string): AccountEvent | null => {
    if (line === "") return dispatch();
    if (line.startsWith("id: ")) {
      const parsed = Number.parseInt(line.slice(4).trim(), 10);
      currentId = Number.isFinite(parsed) ? parsed : 0;
      return null;
    }
    if (line.startsWith("event: ")) {
      currentEvent = line.slice(7).trim();
      return null;
    }
    if (line.startsWith("data: ")) dataLines.push(line.slice(6));
    return null;
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const rawLine of lines) {
      const event = handleLine(rawLine.replace(/\r$/, ""));
      if (event) yield event;
    }
  }

  if (buffer.length > 0) {
    const event = handleLine(buffer.replace(/\r$/, ""));
    if (event) yield event;
  }

  const event = dispatch();
  if (event) yield event;
}
