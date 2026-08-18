import { Navigate, Outlet } from "react-router";
import { useAuth } from "@/hooks/useAuth";
import { canManageUsers } from "@/lib/authz";

export default function AdminRoute() {
  const { user } = useAuth();
  if (!canManageUsers(user?.role)) {
    return <Navigate to="/" replace state={{ forbidden: true }} />;
  }
  return <Outlet />;
}
