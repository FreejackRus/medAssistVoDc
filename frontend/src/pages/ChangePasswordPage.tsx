import { useState } from "react";
import type { FormEvent } from "react";
import { KeyRound, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useAuth } from "@/hooks/useAuth";

export default function ChangePasswordPage() {
  const { changePassword, logout, user } = useAuth();
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [repeatPassword, setRepeatPassword] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (newPassword !== repeatPassword) {
      setError("Новые пароли не совпадают");
      return;
    }
    setIsSubmitting(true);
    setError(null);
    try {
      await changePassword(currentPassword, newPassword);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Не удалось сменить пароль");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-muted/30 p-4">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle>Смена одноразового пароля</CardTitle>
        </CardHeader>
        <CardContent>
          <form className="space-y-4" onSubmit={submit}>
            <p className="text-sm text-muted-foreground">
              Аккаунт: <span className="font-medium text-foreground">{user?.username}</span>
            </p>
            {!user?.must_change_password && (
              <div className="space-y-2">
                <label className="text-sm font-medium" htmlFor="current-password">
                  Текущий пароль
                </label>
                <Input
                  id="current-password"
                  type="password"
                  autoComplete="current-password"
                  value={currentPassword}
                  onChange={(e) => setCurrentPassword(e.target.value)}
                  disabled={isSubmitting}
                />
              </div>
            )}
            <div className="space-y-2">
              <label className="text-sm font-medium" htmlFor="new-password">
                Новый пароль
              </label>
              <Input
                id="new-password"
                type="password"
                autoComplete="new-password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                disabled={isSubmitting}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium" htmlFor="repeat-password">
                Повторите новый пароль
              </label>
              <Input
                id="repeat-password"
                type="password"
                autoComplete="new-password"
                value={repeatPassword}
                onChange={(e) => setRepeatPassword(e.target.value)}
                disabled={isSubmitting}
              />
            </div>
            {error && <p className="text-sm text-destructive">{error}</p>}
            <div className="flex gap-2">
              <Button type="submit" className="flex-1 gap-2" disabled={isSubmitting}>
                {isSubmitting ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <KeyRound className="h-4 w-4" />
                )}
                Сменить пароль
              </Button>
              <Button type="button" variant="outline" onClick={logout} disabled={isSubmitting}>
                Выйти
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
