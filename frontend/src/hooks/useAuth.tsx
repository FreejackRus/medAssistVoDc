import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { apiFetch, getAuthToken, setAuthToken } from "@/lib/api";

export interface AuthUser {
  id: string;
  username: string;
  role: "admin" | "manager" | "user";
  must_change_password: boolean;
  display_name: string | null;
  organization: string | null;
  position: string | null;
  notes: string | null;
  profile_fields: Record<string, unknown>;
  allowed_profile_fields: string[];
  active_sessions: number;
}

interface LoginResponse {
  token: string;
  user: AuthUser;
}

interface AuthContextValue {
  user: AuthUser | null;
  token: string | null;
  isLoading: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  changePassword: (currentPassword: string, newPassword: string) => Promise<void>;
  completeOnboarding: (token: string, newPassword: string) => Promise<void>;
  updateProfile: (body: Partial<Pick<AuthUser, "display_name" | "organization" | "position" | "notes">>) => Promise<void>;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const qc = useQueryClient();
  const [token, setToken] = useState<string | null>(() => getAuthToken());
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isLoading, setIsLoading] = useState(!!token);

  const clearAuth = useCallback(() => {
    setAuthToken(null);
    setToken(null);
    setUser(null);
    qc.clear();
  }, [qc]);

  const refreshUser = useCallback(async () => {
    if (!getAuthToken()) {
      setUser(null);
      setIsLoading(false);
      return;
    }
    try {
      const next = await apiFetch<AuthUser>("/auth/me");
      setUser(next);
    } catch {
      clearAuth();
    } finally {
      setIsLoading(false);
    }
  }, [clearAuth]);

  useEffect(() => {
    void refreshUser();
  }, [refreshUser]);

  const login = useCallback(
    async (username: string, password: string) => {
      const res = await apiFetch<LoginResponse>("/auth/login", {
        method: "POST",
        body: JSON.stringify({ username, password }),
      });
      setAuthToken(res.token);
      setToken(res.token);
      setUser(res.user);
      qc.clear();
    },
    [qc],
  );

  const logout = useCallback(async () => {
    try {
      if (getAuthToken()) {
        await apiFetch("/auth/logout", { method: "POST" });
      }
    } finally {
      clearAuth();
    }
  }, [clearAuth]);

  const changePassword = useCallback(
    async (currentPassword: string, newPassword: string) => {
      const next = await apiFetch<AuthUser>("/auth/change-password", {
        method: "POST",
        body: JSON.stringify({
          current_password: currentPassword || null,
          new_password: newPassword,
        }),
      });
      setUser(next);
      qc.clear();
    },
    [qc],
  );

  const completeOnboarding = useCallback(
    async (onboardingToken: string, newPassword: string) => {
      const res = await apiFetch<LoginResponse>("/auth/complete-onboarding", {
        method: "POST",
        body: JSON.stringify({ token: onboardingToken, new_password: newPassword }),
      });
      setAuthToken(res.token);
      setToken(res.token);
      setUser(res.user);
      qc.clear();
    },
    [qc],
  );

  const updateProfile = useCallback(
    async (
      body: Partial<
        Pick<AuthUser, "display_name" | "organization" | "position" | "notes">
      >,
    ) => {
      const next = await apiFetch<AuthUser>("/auth/profile", {
        method: "PATCH",
        body: JSON.stringify(body),
      });
      setUser(next);
      qc.invalidateQueries({ queryKey: ["admin-users"] });
    },
    [qc],
  );

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      token,
      isLoading,
      login,
      logout,
      changePassword,
      completeOnboarding,
      updateProfile,
      refreshUser,
    }),
    [changePassword, completeOnboarding, isLoading, login, logout, refreshUser, token, updateProfile, user],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside AuthProvider");
  return ctx;
}
