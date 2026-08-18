import { useEffect, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { readAccountEvents } from "@/lib/api";
import { HttpError } from "@/lib/httpError";
import { useAuth } from "@/hooks/useAuth";

export function isTerminalAccountEventError(error: unknown): boolean {
  return error instanceof HttpError && (error.status === 401 || error.status === 403);
}

export function accountEventReconnectDelay(attempt: number, random = Math.random): number {
  const baseDelay = Math.min(1000 * 2 ** attempt, 30_000);
  return Math.round(baseDelay * (0.8 + random() * 0.4));
}

export function useAccountEvents() {
  const qc = useQueryClient();
  const { user, refreshUser } = useAuth();
  const userId = user?.id ?? null;
  const mustChangePassword = user?.must_change_password ?? false;
  const lastEventIdRef = useRef(0);

  useEffect(() => {
    if (!userId || mustChangePassword) return;

    const controller = new AbortController();
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let refreshTimer: ReturnType<typeof setTimeout> | null = null;
    let adminUsersTimer: ReturnType<typeof setTimeout> | null = null;
    let reconnectAttempt = 0;

    const scheduleRefreshUser = () => {
      if (refreshTimer) clearTimeout(refreshTimer);
      refreshTimer = setTimeout(() => {
        refreshTimer = null;
        void refreshUser();
      }, 500);
    };
    const scheduleAdminUsersRefresh = () => {
      if (adminUsersTimer) clearTimeout(adminUsersTimer);
      adminUsersTimer = setTimeout(() => {
        adminUsersTimer = null;
        qc.invalidateQueries({ queryKey: ["admin-users"] });
      }, 500);
    };
    const scheduleReconnect = () => {
      const delay = accountEventReconnectDelay(reconnectAttempt);
      reconnectAttempt += 1;
      reconnectTimer = setTimeout(run, delay);
    };
    const run = async () => {
      try {
        for await (const event of readAccountEvents(
          lastEventIdRef.current,
          controller.signal,
        )) {
          reconnectAttempt = 0;
          if (event.id) lastEventIdRef.current = event.id;

          const documentId =
            typeof event.payload.document_id === "string"
              ? event.payload.document_id
              : undefined;
          const sessionId =
            typeof event.payload.session_id === "string"
              ? event.payload.session_id
              : undefined;
          const algorithmId =
            typeof event.payload.algorithm_id === "string"
              ? event.payload.algorithm_id
              : undefined;

          if (event.event === "documents_changed") {
            qc.invalidateQueries({ queryKey: ["documents"] });
            if (documentId) {
              qc.invalidateQueries({ queryKey: ["algorithm-by-doc", documentId] });
              qc.invalidateQueries({ queryKey: ["sessions", documentId] });
            }
          }

          if (event.event === "sessions_changed") {
            qc.invalidateQueries({ queryKey: ["sessions"] });
            if (documentId) qc.invalidateQueries({ queryKey: ["sessions", documentId] });
          }

          if (event.event === "messages_changed") {
            if (sessionId) qc.invalidateQueries({ queryKey: ["messages", sessionId] });
            qc.invalidateQueries({ queryKey: ["sessions"] });
            if (documentId) qc.invalidateQueries({ queryKey: ["sessions", documentId] });
          }

          if (event.event === "algorithm_changed") {
            if (documentId) {
              qc.invalidateQueries({ queryKey: ["algorithm-by-doc", documentId] });
            }
            if (algorithmId) qc.invalidateQueries({ queryKey: ["algorithm", algorithmId] });
          }

          if (event.event === "presence_changed") {
            scheduleRefreshUser();
          }

          if (event.event === "admin_users_changed") {
            scheduleAdminUsersRefresh();
            scheduleRefreshUser();
          }
        }
        if (!controller.signal.aborted) scheduleReconnect();
      } catch (error) {
        if (!controller.signal.aborted) {
          if (isTerminalAccountEventError(error)) {
            return;
          }
          scheduleReconnect();
        }
      }
    };

    void run();

    return () => {
      controller.abort();
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (refreshTimer) clearTimeout(refreshTimer);
      if (adminUsersTimer) clearTimeout(adminUsersTimer);
    };
  }, [qc, refreshUser, userId, mustChangePassword]);
}
