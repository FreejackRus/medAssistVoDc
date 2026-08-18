import { useState } from "react";
import { NavLink, Outlet } from "react-router";
import {
  FileText,
  MessageSquare,
  Calculator,
  BookOpen,
  Users,
  Activity,
  LogOut,
  Menu,
  X,
  AlertTriangle,
  UserCircle,
} from "lucide-react";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";
import ThemeToggle from "@/components/shared/ThemeToggle";
import { useAuth } from "@/hooks/useAuth";
import { useAccountEvents } from "@/hooks/useAccountEvents";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ProfileForm } from "@/components/profile/ProfileForm";

const baseNavItems = [
  { to: "/", icon: FileText, label: "Документы" },
  { to: "/chat", icon: MessageSquare, label: "Чат" },
  { to: "/calculators", icon: Calculator, label: "Калькуляторы" },
  { to: "/clinical-recommendations", icon: BookOpen, label: "Рекомендации" },
];

function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  const { user, logout } = useAuth();
  const [profileOpen, setProfileOpen] = useState(false);
  const navItems =
    user?.role === "admin" || user?.role === "manager"
      ? [
          ...baseNavItems,
          { to: "/admin/users", icon: Users, label: "Пользователи" },
          { to: "/admin/monitoring", icon: Activity, label: "Мониторинг" },
        ]
      : baseNavItems;

  return (
    <>
      <div className="p-6">
        <h1 className="text-lg font-semibold">МедАссистент</h1>
        <p className="text-sm text-muted-foreground">Информационная поддержка</p>
      </div>
      <Separator />
      <nav className="flex-1 p-4 space-y-1">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            onClick={onNavigate}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors ${
                isActive
                  ? "bg-sidebar-accent text-sidebar-accent-foreground font-medium"
                  : "text-sidebar-foreground hover:bg-sidebar-accent/50"
              }`
            }
          >
            <Icon className="h-4 w-4" />
            {label}
          </NavLink>
        ))}
      </nav>
      <Separator />
      <div className="flex items-center justify-between p-3">
        <span className="text-xs text-muted-foreground">Оформление</span>
        <ThemeToggle />
      </div>
      <Separator />
      <div className="space-y-2 p-3">
        <button
          type="button"
          className="flex w-full items-center gap-3 rounded-lg px-2 py-2 text-left transition-colors hover:bg-sidebar-accent/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          onClick={() => setProfileOpen(true)}
        >
          <UserCircle className="h-8 w-8 shrink-0 text-muted-foreground" />
          <span className="min-w-0 flex-1">
            <span className="block truncate text-sm font-medium">{user?.username}</span>
            <span className="block text-xs text-muted-foreground">
              {user?.role === "admin"
                ? "Администратор"
                : user?.role === "manager"
                  ? "Менеджер"
                  : "Пользователь"}
              {user && user.active_sessions > 1 ? ` · активных входов: ${user.active_sessions}` : ""}
            </span>
          </span>
        </button>
        <Dialog open={profileOpen} onOpenChange={setProfileOpen}>
          <DialogContent className="sm:max-w-xl">
            <DialogHeader>
              <DialogTitle>{user?.username}</DialogTitle>
              <DialogDescription>
                Данные профиля и текущая учетная запись. Доступность полей для изменения
                настраивается администратором.
              </DialogDescription>
            </DialogHeader>
            <ProfileForm />
            <DialogFooter>
              <Button variant="outline" className="gap-2" onClick={() => void logout()}>
                <LogOut className="h-4 w-4" />
                Выйти
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </>
  );
}

export default function Layout() {
  const [mobileOpen, setMobileOpen] = useState(false);
  useAccountEvents();

  return (
    <div className="flex h-screen">
      {/* Desktop sidebar */}
      <aside className="hidden md:flex w-64 flex-col border-r bg-sidebar">
        <SidebarContent />
      </aside>

      {/* Mobile overlay */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 md:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Mobile sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-50 w-64 flex-col border-r bg-sidebar transition-transform md:hidden ${
          mobileOpen ? "flex translate-x-0" : "-translate-x-full"
        }`}
      >
        <SidebarContent onNavigate={() => setMobileOpen(false)} />
      </aside>

      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Mobile header */}
        <header className="flex items-center gap-3 border-b p-3 md:hidden">
          <Button
            variant="ghost"
            size="icon"
            title="Меню"
            onClick={() => setMobileOpen((v) => !v)}
          >
            {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </Button>
          <span className="text-sm font-medium">МедАссистент</span>
          <div className="ml-auto">
            <ThemeToggle />
          </div>
        </header>
        <div className="border-b bg-muted/50 px-4 py-2 text-xs text-muted-foreground">
          <div className="flex items-start gap-2">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>
              Система предназначена для информационной поддержки и не заменяет
              клиническое решение врача.
            </span>
          </div>
        </div>
        <main className="flex-1 overflow-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
