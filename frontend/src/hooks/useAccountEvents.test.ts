import { describe, expect, it } from "vitest";
import {
  accountEventReconnectDelay,
  isTerminalAccountEventError,
} from "@/hooks/useAccountEvents";
import { HttpError } from "@/lib/httpError";

describe("account event reconnect policy", () => {
  it("stops reconnecting for authentication and authorization failures", () => {
    expect(isTerminalAccountEventError(new HttpError(401, "", "expired"))).toBe(true);
    expect(isTerminalAccountEventError(new HttpError(403, "", "forbidden"))).toBe(true);
    expect(isTerminalAccountEventError(new HttpError(500, "", "server"))).toBe(false);
    expect(isTerminalAccountEventError(new TypeError("network"))).toBe(false);
  });

  it("uses capped exponential backoff with jitter", () => {
    expect(accountEventReconnectDelay(0, () => 0.5)).toBe(1000);
    expect(accountEventReconnectDelay(3, () => 0.5)).toBe(8000);
    expect(accountEventReconnectDelay(10, () => 0.5)).toBe(30_000);
    expect(accountEventReconnectDelay(0, () => 0)).toBe(800);
    expect(accountEventReconnectDelay(0, () => 1)).toBe(1200);
  });
});
