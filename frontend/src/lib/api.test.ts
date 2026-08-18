import { afterEach, describe, expect, it, vi } from "vitest";
import { apiFetch, readAccountEvents, setUnauthorizedHandler } from "@/lib/api";

afterEach(() => {
  vi.unstubAllGlobals();
  setUnauthorizedHandler(null);
});

describe("API authentication", () => {
  it("sends requests with cookie credentials and no bearer token", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await apiFetch("/example");

    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(init.credentials).toBe("include");
    expect(new Headers(init.headers).has("Authorization")).toBe(false);
  });

  it("reports 401 globally and preserves its status", async () => {
    const onUnauthorized = vi.fn();
    setUnauthorizedHandler(onUnauthorized);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ error: "Сессия истекла" }), { status: 401 }),
      ),
    );

    await expect(apiFetch("/private")).rejects.toMatchObject({
      status: 401,
      message: "Сессия истекла",
    });
    expect(onUnauthorized).toHaveBeenCalledOnce();
  });

  it("does not clear authentication on 403", async () => {
    const onUnauthorized = vi.fn();
    setUnauthorizedHandler(onUnauthorized);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ error: "Недостаточно прав" }), { status: 403 }),
      ),
    );

    await expect(apiFetch("/admin")).rejects.toMatchObject({ status: 403 });
    expect(onUnauthorized).not.toHaveBeenCalled();
  });

  it("propagates auth status from SSE connection failures", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ error: "Требуется вход" }), { status: 401 }),
      ),
    );

    await expect(readAccountEvents().next()).rejects.toMatchObject({ status: 401 });
  });
});
