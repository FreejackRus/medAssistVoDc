import { Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useThemeStore } from "@/stores/theme";

export default function ThemeToggle() {
  const theme = useThemeStore((s) => s.theme);
  const toggle = useThemeStore((s) => s.toggle);
  const Icon = theme === "dark" ? Sun : Moon;
  const title = theme === "dark" ? "Светлая тема" : "Тёмная тема";

  return (
    <Button variant="ghost" size="icon" onClick={toggle} title={title}>
      <Icon className="h-5 w-5" />
    </Button>
  );
}
