export type UserRole = "admin" | "manager" | "user";

export function canManageUsers(role: UserRole | null | undefined): boolean {
  return role === "admin" || role === "manager";
}
