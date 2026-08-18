import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";

export interface AdminUser {
  id: string;
  username: string;
  role: "admin" | "manager" | "user";
  must_change_password: boolean;
  created_by: string | null;
  creator_username: string | null;
  display_name: string | null;
  organization: string | null;
  position: string | null;
  notes: string | null;
  profile_fields: string;
  allowed_profile_fields: string;
  onboarding_expires_at: string | null;
  created_at: string;
  updated_at: string;
  last_login_at: string | null;
  active_sessions: number;
  documents_count: number | null;
  algorithm_generations_count: number | null;
  chat_generations_count: number | null;
  documents_with_algorithm_count: number | null;
  documents_with_chat_count: number | null;
}

interface UserWithTemporaryPassword {
  user: AdminUser;
  temporary_password: string;
  onboarding_url: string;
}

interface ResetPasswordResponse {
  temporary_password: string;
  onboarding_url: string;
}

export interface UpdateUserPayload {
  display_name: string | null;
  organization: string | null;
  position: string | null;
  notes: string | null;
  allowed_profile_fields: string[];
}

export function useAdminUsers() {
  return useQuery<AdminUser[]>({
    queryKey: ["admin-users"],
    queryFn: () => apiFetch("/admin/users"),
  });
}

export function useCreateUser() {
  const qc = useQueryClient();
  return useMutation<
    UserWithTemporaryPassword,
    Error,
    { username: string; role: "admin" | "manager" | "user" }
  >({
    mutationFn: (body) =>
      apiFetch("/admin/users", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["admin-users"] }),
  });
}

export function useResetUserPassword() {
  const qc = useQueryClient();
  return useMutation<ResetPasswordResponse, Error, string>({
    mutationFn: (id) =>
      apiFetch(`/admin/users/${id}/reset-password`, { method: "POST" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["admin-users"] }),
  });
}

export function useUpdateUser() {
  const qc = useQueryClient();
  return useMutation<AdminUser, Error, { id: string; body: UpdateUserPayload }>({
    mutationFn: ({ id, body }) =>
      apiFetch(`/admin/users/${id}`, {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["admin-users"] }),
  });
}

export function useDeleteUser() {
  const qc = useQueryClient();
  return useMutation<unknown, Error, string>({
    mutationFn: (id) => apiFetch(`/admin/users/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["admin-users"] }),
  });
}
