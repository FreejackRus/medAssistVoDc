import { useMemo, useState } from "react";
import type { FormEvent } from "react";
import {
  Copy,
  CornerDownRight,
  Edit2,
  KeyRound,
  LayoutGrid,
  Loader2,
  Save,
  Table2,
  Trash2,
  UserPlus,
  X,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import {
  useAdminUsers,
  useCreateUser,
  useDeleteUser,
  useResetUserPassword,
  useUpdateUser,
  type AdminUser,
} from "@/hooks/useAdminUsers";
import { useAuth } from "@/hooks/useAuth";
import { useToast } from "@/components/ui/toast";

const roleLabels = {
  admin: "Администратор",
  manager: "Менеджер",
  user: "Пользователь",
} as const;

const editableFields = [
  ["display_name", "ФИО"],
  ["organization", "Организация"],
  ["position", "Должность"],
  ["notes", "Заметки"],
] as const;

type ManagedRole = "admin" | "manager" | "user";
type ViewMode = "cards" | "table";

interface AccessPayload {
  username: string;
  password: string;
  onboardingUrl: string;
}

interface EditState {
  display_name: string;
  organization: string;
  position: string;
  notes: string;
  allowed_profile_fields: string[];
}

export default function AdminUsersPage() {
  const { user } = useAuth();
  const { data: users, isLoading } = useAdminUsers();
  const createUser = useCreateUser();
  const resetPassword = useResetUserPassword();
  const deleteUser = useDeleteUser();
  const updateUser = useUpdateUser();
  const { toast } = useToast();
  const [username, setUsername] = useState("");
  const [role, setRole] = useState<ManagedRole>("user");
  const [access, setAccess] = useState<AccessPayload | null>(null);
  const [query, setQuery] = useState("");
  const [roleFilter, setRoleFilter] = useState<"all" | ManagedRole>("all");
  const [viewMode, setViewMode] = useState<ViewMode>("table");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [edit, setEdit] = useState<EditState | null>(null);

  const roleOptions: ManagedRole[] =
    user?.role === "admin" ? ["user", "manager", "admin"] : ["user"];

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return (users ?? []).filter((item) => {
      if (roleFilter !== "all" && item.role !== roleFilter) return false;
      if (!q) return true;
      return [
        item.username,
        item.display_name,
        item.organization,
        item.position,
        item.creator_username,
      ]
        .filter(Boolean)
        .some((value) => value!.toLowerCase().includes(q));
    });
  }, [query, roleFilter, users]);

  const usersById = useMemo(
    () => new Map((users ?? []).map((item) => [item.id, item])),
    [users],
  );

  const managedCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const item of users ?? []) {
      if (!item.created_by) continue;
      const creator = usersById.get(item.created_by);
      if (creator?.role !== "manager") continue;
      counts.set(item.created_by, (counts.get(item.created_by) ?? 0) + 1);
    }
    return counts;
  }, [users, usersById]);

  const sortedFiltered = useMemo(() => {
    const userSortKey = (item: AdminUser): [number, string, number, string] => {
      if (item.role === "admin") {
        return [0, item.username, 0, item.username];
      }

      if (item.role === "manager") {
        return [1, item.username, 0, item.username];
      }

      const creator = item.created_by ? usersById.get(item.created_by) : null;
      if (creator?.role === "manager") {
        return [1, creator.username, 1, item.username];
      }

      return [2, item.username, 0, item.username];
    };

    return [...filtered].sort((a, b) => {
      const aKey = userSortKey(a);
      const bKey = userSortKey(b);
      return (
        aKey[0] - bKey[0] ||
        aKey[1].localeCompare(bKey[1], "ru") ||
        aKey[2] - bKey[2] ||
        aKey[3].localeCompare(bKey[3], "ru")
      );
    });
  }, [filtered, usersById]);

  const copyText = async (text: string, label: string) => {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else {
        const area = document.createElement("textarea");
        area.value = text;
        area.style.position = "fixed";
        area.style.left = "-9999px";
        document.body.appendChild(area);
        area.focus();
        area.select();
        document.execCommand("copy");
        document.body.removeChild(area);
      }
      toast(label, "success");
    } catch {
      toast("Не удалось скопировать", "error");
    }
  };

  const fullOnboardingUrl = (path: string) => `${window.location.origin}${path}`;

  const submitCreate = (e: FormEvent) => {
    e.preventDefault();
    createUser.mutate(
      { username, role },
      {
        onSuccess: (res) => {
          setUsername("");
          setRole("user");
          setAccess({
            username: res.user.username,
            password: res.temporary_password,
            onboardingUrl: fullOnboardingUrl(res.onboarding_url),
          });
        },
        onError: (err) => toast(err.message, "error"),
      },
    );
  };

  const handleReset = (target: AdminUser) => {
    resetPassword.mutate(target.id, {
      onSuccess: (res) =>
        setAccess({
          username: target.username,
          password: res.temporary_password,
          onboardingUrl: fullOnboardingUrl(res.onboarding_url),
        }),
      onError: (err) => toast(err.message, "error"),
    });
  };

  const handleDelete = (target: AdminUser) => {
    if (!confirm(`Удалить учетную запись ${target.username}?`)) return;
    deleteUser.mutate(target.id, {
      onError: (err) => toast(err.message, "error"),
    });
  };

  const startEdit = (target: AdminUser) => {
    setEditingId(target.id);
    setEdit({
      display_name: target.display_name ?? "",
      organization: target.organization ?? "",
      position: target.position ?? "",
      notes: target.notes ?? "",
      allowed_profile_fields: parseAllowed(target.allowed_profile_fields),
    });
  };

  const saveEdit = (target: AdminUser) => {
    if (!edit) return;
    updateUser.mutate(
      {
        id: target.id,
        body: {
          display_name: edit.display_name || null,
          organization: edit.organization || null,
          position: edit.position || null,
          notes: edit.notes || null,
          allowed_profile_fields: edit.allowed_profile_fields,
        },
      },
      {
        onSuccess: () => {
          setEditingId(null);
          setEdit(null);
        },
        onError: (err) => toast(err.message, "error"),
      },
    );
  };

  const toggleAllowed = (field: string) => {
    setEdit((prev) => {
      if (!prev) return prev;
      const current = new Set(prev.allowed_profile_fields);
      if (current.has(field)) current.delete(field);
      else current.add(field);
      return { ...prev, allowed_profile_fields: Array.from(current) };
    });
  };

  return (
    <div className="space-y-6 p-6">
      <div>
        <h2 className="text-2xl font-bold">Пользователи</h2>
        <p className="text-sm text-muted-foreground">
          Управление учетными записями, приглашениями и карточками пользователей.
        </p>
      </div>

      <Card className="overflow-visible">
        <CardHeader>
          <CardTitle className="text-base">Создать пользователя</CardTitle>
        </CardHeader>
        <CardContent>
          <form className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_12rem_auto] sm:items-end" onSubmit={submitCreate}>
            <Input
              className="h-10"
              placeholder="Логин"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              disabled={createUser.isPending}
            />
            <Select
              triggerClassName="h-10"
              value={role}
              options={roleOptions.map((value) => ({
                value,
                label: roleLabels[value],
              }))}
              onValueChange={(value) => setRole(value as ManagedRole)}
              disabled={createUser.isPending}
            />
            <Button type="submit" className="h-10 gap-2" disabled={createUser.isPending}>
              {createUser.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <UserPlus className="h-4 w-4" />
              )}
              Создать
            </Button>
          </form>
          {access && (
            <div className="mt-4 space-y-3 rounded-md border bg-muted/50 p-3">
              <p className="text-sm font-medium">Доступ для {access.username}</p>
              <AccessRow
                label="Одноразовый пароль"
                value={access.password}
                onCopy={() => copyText(access.password, "Одноразовый пароль скопирован")}
              />
              <AccessRow
                label="Ссылка приглашения"
                value={access.onboardingUrl}
                onCopy={() => copyText(access.onboardingUrl, "Ссылка приглашения скопирована")}
              />
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_12rem_auto] lg:items-center">
        <Input
          className="h-10"
          placeholder="Поиск по логину, ФИО, организации, создателю..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <Select
          triggerClassName="h-10"
          value={roleFilter}
          options={[
            { value: "all", label: "Все роли" },
            { value: "admin", label: "Администраторы" },
            { value: "manager", label: "Менеджеры" },
            { value: "user", label: "Пользователи" },
          ]}
          onValueChange={(value) => setRoleFilter(value as "all" | ManagedRole)}
        />
        <div className="inline-flex h-10 w-fit rounded-md border bg-background p-1">
          <Button
            type="button"
            variant={viewMode === "table" ? "secondary" : "ghost"}
            size="sm"
            className="h-8 gap-2 px-3"
            onClick={() => setViewMode("table")}
          >
            <Table2 className="h-4 w-4" />
            Таблица
          </Button>
          <Button
            type="button"
            variant={viewMode === "cards" ? "secondary" : "ghost"}
            size="sm"
            className="h-8 gap-2 px-3"
            onClick={() => setViewMode("cards")}
          >
            <LayoutGrid className="h-4 w-4" />
            Карточки
          </Button>
        </div>
      </div>

      {isLoading ? (
        <div className="flex justify-center py-8">
          <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
        </div>
      ) : sortedFiltered.length === 0 ? (
        <div className="rounded-md border bg-muted/30 p-6 text-center text-sm text-muted-foreground">
          Пользователи по выбранным фильтрам не найдены.
        </div>
      ) : viewMode === "table" ? (
        <UsersTable
          users={sortedFiltered}
          usersById={usersById}
          managedCounts={managedCounts}
          currentUserId={user?.id ?? null}
          onEdit={(item) => {
            startEdit(item);
            setViewMode("cards");
          }}
          onReset={handleReset}
          onDelete={handleDelete}
          resetPending={resetPassword.isPending}
          deletePending={deleteUser.isPending}
        />
      ) : (
        <div className="grid gap-3 xl:grid-cols-2">
          {sortedFiltered.map((item) => {
            const isEditing = editingId === item.id && !!edit;
            const creator = item.created_by ? (usersById.get(item.created_by) ?? null) : null;
            return (
              <Card key={item.id}>
                <CardContent className="space-y-4 p-4">
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-start">
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <p className="font-medium">{item.username}</p>
                        <Badge variant={item.role === "admin" ? "default" : "secondary"}>
                          {roleLabels[item.role]}
                        </Badge>
                        {item.must_change_password && (
                          <Badge variant="destructive">Требует смены пароля</Badge>
                        )}
                        {item.active_sessions > 0 && (
                          <Badge variant="secondary">Активных входов: {item.active_sessions}</Badge>
                        )}
                      </div>
                      <UserRelation
                        item={item}
                        creator={creator}
                        managedCount={managedCounts.get(item.id) ?? 0}
                      />
                      <p className="mt-1 text-xs text-muted-foreground">
                        Последний вход: {item.last_login_at ?? "не выполнялся"}
                      </p>
                    </div>
                    <div className="flex gap-2">
                      {isEditing ? (
                        <>
                          <Button
                            size="sm"
                            className="gap-2"
                            onClick={() => saveEdit(item)}
                            disabled={updateUser.isPending}
                          >
                            <Save className="h-4 w-4" />
                            Сохранить
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => {
                              setEditingId(null);
                              setEdit(null);
                            }}
                          >
                            <X className="h-4 w-4" />
                          </Button>
                        </>
                      ) : (
                        <Button
                          variant="outline"
                          size="sm"
                          className="gap-2"
                          onClick={() => startEdit(item)}
                        >
                          <Edit2 className="h-4 w-4" />
                          Карточка
                        </Button>
                      )}
                      <Button
                        variant="outline"
                        size="sm"
                        className="gap-2"
                        onClick={() => handleReset(item)}
                        disabled={resetPassword.isPending}
                      >
                        <KeyRound className="h-4 w-4" />
                        Сбросить
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        title="Удалить"
                        onClick={() => handleDelete(item)}
                        disabled={deleteUser.isPending || item.id === user?.id}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>

                  {isEditing && edit ? (
                    <div className="grid gap-3 md:grid-cols-2">
                      <LabeledInput
                        label="ФИО"
                        value={edit.display_name}
                        onChange={(value) => setEdit({ ...edit, display_name: value })}
                      />
                      <LabeledInput
                        label="Организация"
                        value={edit.organization}
                        onChange={(value) => setEdit({ ...edit, organization: value })}
                      />
                      <LabeledInput
                        label="Должность"
                        value={edit.position}
                        onChange={(value) => setEdit({ ...edit, position: value })}
                      />
                      <div className="space-y-2 md:col-span-2">
                        <label className="text-sm font-medium">Заметки</label>
                        <textarea
                          className="min-h-20 w-full rounded-md border bg-background px-3 py-2 text-sm"
                          value={edit.notes}
                          onChange={(e) => setEdit({ ...edit, notes: e.target.value })}
                        />
                      </div>
                      <div className="space-y-2 md:col-span-2">
                        <p className="text-sm font-medium">Поля, которые пользователь может менять сам</p>
                        <div className="flex flex-wrap gap-3">
                          {editableFields.map(([field, label]) => (
                            <label
                              key={field}
                              className="inline-flex cursor-pointer items-center gap-2 text-sm"
                            >
                              <Checkbox
                                checked={edit.allowed_profile_fields.includes(field)}
                                onChange={() => toggleAllowed(field)}
                              />
                              {label}
                            </label>
                          ))}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <div className="grid gap-2 text-sm md:grid-cols-2">
                        <Info label="ФИО" value={item.display_name} />
                        <Info label="Организация" value={item.organization} />
                        <Info label="Должность" value={item.position} />
                        <Info label="Заметки" value={item.notes} className="md:col-span-2" />
                      </div>
                      {user?.role === "admin" && (
                        <div className="grid gap-2 border-t pt-3 text-sm sm:grid-cols-2 xl:grid-cols-3">
                          <Stat label="Документы" value={item.documents_count} />
                          <Stat
                            label="Генерации алгоритмов"
                            value={item.algorithm_generations_count}
                          />
                          <Stat label="Ответы в чатах" value={item.chat_generations_count} />
                          <Stat
                            label="Документы с алгоритмом"
                            value={item.documents_with_algorithm_count}
                          />
                          <Stat
                            label="Документы с диалогом"
                            value={item.documents_with_chat_count}
                          />
                        </div>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}

function UsersTable({
  users,
  usersById,
  managedCounts,
  currentUserId,
  onEdit,
  onReset,
  onDelete,
  resetPending,
  deletePending,
}: {
  users: AdminUser[];
  usersById: Map<string, AdminUser>;
  managedCounts: Map<string, number>;
  currentUserId: string | null;
  onEdit: (target: AdminUser) => void;
  onReset: (target: AdminUser) => void;
  onDelete: (target: AdminUser) => void;
  resetPending: boolean;
  deletePending: boolean;
}) {
  return (
    <div className="overflow-hidden rounded-lg border bg-card">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[980px] text-left text-sm">
          <thead className="border-b bg-muted/40 text-xs text-muted-foreground">
            <tr>
              <th className="px-4 py-3 font-medium">Учетная запись</th>
              <th className="px-4 py-3 font-medium">Роль</th>
              <th className="px-4 py-3 font-medium">Связь</th>
              <th className="px-4 py-3 font-medium">Статус</th>
              <th className="px-4 py-3 font-medium">Использование</th>
              <th className="px-4 py-3 font-medium">Последний вход</th>
              <th className="px-4 py-3 text-right font-medium">Действия</th>
            </tr>
          </thead>
          <tbody>
            {users.map((item) => {
              const creator = item.created_by ? (usersById.get(item.created_by) ?? null) : null;
              const isManagedByManager = creator?.role === "manager";
              return (
                <tr key={item.id} className="border-b last:border-0">
                  <td className="px-4 py-3">
                    <div className={isManagedByManager ? "flex items-start gap-2 pl-5" : ""}>
                      {isManagedByManager && (
                        <CornerDownRight className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
                      )}
                      <div className="min-w-0">
                        <p className="truncate font-medium">{item.username}</p>
                        <p className="truncate text-xs text-muted-foreground">
                          {item.display_name || item.organization || item.position || "Карточка не заполнена"}
                        </p>
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <Badge variant={item.role === "admin" ? "default" : "secondary"}>
                      {roleLabels[item.role]}
                    </Badge>
                  </td>
                  <td className="px-4 py-3 text-xs text-muted-foreground">
                    {isManagedByManager ? (
                      <span className="font-medium text-foreground">Менеджер: {creator.username}</span>
                    ) : (
                      <span>Создал: {item.creator_username ?? "система"}</span>
                    )}
                    {item.role === "manager" && (
                      <span className="mt-1 block">Пользователей: {managedCounts.get(item.id) ?? 0}</span>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex flex-wrap gap-1">
                      {item.must_change_password && (
                        <Badge variant="destructive">Требует смены пароля</Badge>
                      )}
                      {item.active_sessions > 0 ? (
                        <Badge variant="secondary">Входов: {item.active_sessions}</Badge>
                      ) : (
                        <Badge variant="outline">Нет активных входов</Badge>
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-3 text-xs text-muted-foreground">
                    <span className="block">Документы: {item.documents_count ?? "—"}</span>
                    <span className="block">
                      Алгоритмы: {item.algorithm_generations_count ?? "—"} · Чаты:{" "}
                      {item.chat_generations_count ?? "—"}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-xs text-muted-foreground">
                    {item.last_login_at ?? "не выполнялся"}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex justify-end gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        className="gap-2"
                        onClick={() => onEdit(item)}
                      >
                        <Edit2 className="h-4 w-4" />
                        Карточка
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        className="gap-2"
                        onClick={() => onReset(item)}
                        disabled={resetPending}
                      >
                        <KeyRound className="h-4 w-4" />
                        Сбросить
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        title="Удалить"
                        onClick={() => onDelete(item)}
                        disabled={deletePending || item.id === currentUserId}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function UserRelation({
  item,
  creator,
  managedCount,
}: {
  item: AdminUser;
  creator: AdminUser | null;
  managedCount: number;
}) {
  const isManagedByManager = creator?.role === "manager";
  return (
    <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
      {isManagedByManager ? (
        <Badge variant="outline" className="gap-1">
          <CornerDownRight className="h-3 w-3" />
          Менеджер: {creator.username}
        </Badge>
      ) : (
        <span>Создал: {item.creator_username ?? "система"}</span>
      )}
      {item.role === "manager" && (
        <Badge variant="outline">Пользователей: {managedCount}</Badge>
      )}
    </div>
  );
}

function AccessRow({
  label,
  value,
  onCopy,
}: {
  label: string;
  value: string;
  onCopy: () => void;
}) {
  return (
    <div className="space-y-1">
      <p className="text-xs text-muted-foreground">{label}</p>
      <div className="flex min-w-0 items-center gap-2">
        <code className="min-w-0 flex-1 overflow-hidden text-ellipsis rounded bg-background px-2 py-1 text-sm">
          {value}
        </code>
        <Button variant="outline" size="sm" className="gap-2" onClick={onCopy}>
          <Copy className="h-4 w-4" />
          Скопировать
        </Button>
      </div>
    </div>
  );
}

function LabeledInput({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium">{label}</label>
      <Input value={value} onChange={(e) => onChange(e.target.value)} />
    </div>
  );
}

function Info({
  label,
  value,
  className,
}: {
  label: string;
  value: string | null;
  className?: string;
}) {
  return (
    <div className={className}>
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="min-h-5 whitespace-pre-wrap">{value || "—"}</p>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number | null }) {
  return (
    <div className="rounded-md bg-muted/50 px-3 py-2">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-lg font-semibold leading-6">{value ?? "—"}</p>
    </div>
  );
}

function parseAllowed(raw: string): string[] {
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((v) => typeof v === "string") : [];
  } catch {
    return [];
  }
}
