import { lazy, Suspense } from "react";
import { Routes, Route, Navigate } from "react-router";
import { useLocation } from "react-router";
import { Loader2 } from "lucide-react";
import Layout from "@/components/layout/Layout";
import AdminRoute from "@/components/routing/AdminRoute";
import { useAuth } from "@/hooks/useAuth";

const HomePage = lazy(() => import("@/pages/HomePage"));
const ChatPage = lazy(() => import("@/pages/ChatPage"));
const CalculatorsPage = lazy(() => import("@/pages/CalculatorsPage"));
const CalculatorGroupPage = lazy(() => import("@/pages/CalculatorGroupPage"));
const ClinicalRecsPage = lazy(() => import("@/pages/ClinicalRecsPage"));
const AdminUsersPage = lazy(() => import("@/pages/AdminUsersPage"));
const MonitoringPage = lazy(() => import("@/pages/MonitoringPage"));
const ProfilePage = lazy(() => import("@/pages/ProfilePage"));
const LoginPage = lazy(() => import("@/pages/LoginPage"));
const ChangePasswordPage = lazy(() => import("@/pages/ChangePasswordPage"));
const WelcomePage = lazy(() => import("@/pages/WelcomePage"));
const NotFoundPage = lazy(() => import("@/pages/NotFoundPage"));

function PageLoader() {
  return (
    <div className="flex h-full items-center justify-center">
      <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
    </div>
  );
}

export default function App() {
  const { user, isLoading } = useAuth();
  const location = useLocation();

  if (isLoading) return <PageLoader />;

  if (!user) {
    return (
      <Suspense fallback={<PageLoader />}>
        {location.pathname === "/welcome" ? <WelcomePage /> : <LoginPage />}
      </Suspense>
    );
  }

  if (user.must_change_password) {
    return (
      <Suspense fallback={<PageLoader />}>
        <ChangePasswordPage />
      </Suspense>
    );
  }

  if (location.pathname === "/welcome") {
    return <Navigate to="/" replace />;
  }

  return (
    <Suspense fallback={<PageLoader />}>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<HomePage />} />
          <Route path="chat" element={<ChatPage />} />
          <Route path="calculators" element={<CalculatorsPage />} />
          <Route path="calculators/:groupId" element={<CalculatorGroupPage />} />
          <Route
            path="calculators/:groupId/:calculatorId"
            element={<CalculatorGroupPage />}
          />
          <Route path="clinical-recommendations" element={<ClinicalRecsPage />} />
          <Route path="profile" element={<ProfilePage />} />
          <Route element={<AdminRoute />}>
            <Route path="admin/users" element={<AdminUsersPage />} />
            <Route path="admin/monitoring" element={<MonitoringPage />} />
          </Route>
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
    </Suspense>
  );
}
