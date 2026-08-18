import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useAuth } from "@/hooks/useAuth";
import { ProfileForm } from "@/components/profile/ProfileForm";

export default function ProfilePage() {
  const { user } = useAuth();

  return (
    <div className="space-y-6 p-6">
      <div>
        <h2 className="text-2xl font-bold">Профиль</h2>
        <p className="text-sm text-muted-foreground">
          Доступность полей для изменения настраивается администратором.
        </p>
      </div>
      <Card className="max-w-2xl">
        <CardHeader>
          <CardTitle className="text-base">{user?.username}</CardTitle>
        </CardHeader>
        <CardContent>
          <ProfileForm />
        </CardContent>
      </Card>
    </div>
  );
}
