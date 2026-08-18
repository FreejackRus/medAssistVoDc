import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { Save } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useAuth } from "@/hooks/useAuth";
import { useToast } from "@/components/ui/toast";

const fields = [
  ["display_name", "ФИО"],
  ["organization", "Организация"],
  ["position", "Должность"],
  ["notes", "Заметки"],
] as const;

interface ProfileFormProps {
  onSaved?: () => void;
}

export function ProfileForm({ onSaved }: ProfileFormProps) {
  const { user, updateProfile } = useAuth();
  const { toast } = useToast();
  const [values, setValues] = useState({
    display_name: user?.display_name ?? "",
    organization: user?.organization ?? "",
    position: user?.position ?? "",
    notes: user?.notes ?? "",
  });
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    setValues({
      display_name: user?.display_name ?? "",
      organization: user?.organization ?? "",
      position: user?.position ?? "",
      notes: user?.notes ?? "",
    });
  }, [user?.display_name, user?.id, user?.notes, user?.organization, user?.position]);

  const allowed = new Set(user?.allowed_profile_fields ?? []);
  const canEdit = (field: string) => user?.role === "admin" || allowed.has(field);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setIsSaving(true);
    try {
      await updateProfile(values);
      toast("Профиль обновлен", "success");
      onSaved?.();
    } catch (err) {
      toast(err instanceof Error ? err.message : "Не удалось сохранить профиль", "error");
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <form className="space-y-4" onSubmit={submit}>
      {fields.map(([key, label]) => (
        <div key={key} className="space-y-2">
          <label className="text-sm font-medium" htmlFor={`profile-${key}`}>
            {label}
          </label>
          {key === "notes" ? (
            <textarea
              id={`profile-${key}`}
              className="min-h-24 w-full rounded-md border bg-background px-3 py-2 text-sm disabled:opacity-60"
              value={values[key]}
              disabled={!canEdit(key) || isSaving}
              onChange={(e) => setValues((prev) => ({ ...prev, [key]: e.target.value }))}
            />
          ) : (
            <Input
              id={`profile-${key}`}
              value={values[key]}
              disabled={!canEdit(key) || isSaving}
              onChange={(e) => setValues((prev) => ({ ...prev, [key]: e.target.value }))}
            />
          )}
        </div>
      ))}
      <Button type="submit" className="gap-2" disabled={isSaving}>
        <Save className="h-4 w-4" />
        Сохранить
      </Button>
    </form>
  );
}
