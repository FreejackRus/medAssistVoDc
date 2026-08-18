import { useEffect, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { readAccountEvents } from "@/lib/api";
import { useAuth } from "@/hooks/useAuth";

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
    let monitoringTimer: ReturnType<typeof setTimeout> | null = null;

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
    const scheduleMonitoringRefresh = () => {
      if (monitoringTimer) clearTimeout(monitoringTimer);
      monitoringTimer = setTimeout(() => {
        monitoringTimer = null;
        qc.invalidateQueries({ queryKey: ["monitoring"] });
      }, 500);
    };

    const run = async () => {
      try {
        for await (const event of readAccountEvents(
          lastEventIdRef.current,
          controller.signal,
        )) {
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
            scheduleMonitoringRefresh();
          }

          if (event.event === "algorithm_changed") {
            if (documentId) {
              qc.invalidateQueries({ queryKey: ["algorithm-by-doc", documentId] });
            }
            if (algorithmId) qc.invalidateQueries({ queryKey: ["algorithm", algorithmId] });
            scheduleMonitoringRefresh();
          }

          if (event.event === "presence_changed") {
            scheduleRefreshUser();
          }

          if (event.event === "admin_users_changed") {
            scheduleAdminUsersRefresh();
            scheduleMonitoringRefresh();
            scheduleRefreshUser();
          }
        }
      } catch {
        if (!controller.signal.aborted) {
          reconnectTimer = setTimeout(run, 1500);
        }
      }
    };

    void run();

    return () => {
      controller.abort();
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (refreshTimer) clearTimeout(refreshTimer);
      if (adminUsersTimer) clearTimeout(adminUsersTimer);
      if (monitoringTimer) clearTimeout(monitoringTimer);
    };
  }, [qc, refreshUser, userId, mustChangePassword]);
}
