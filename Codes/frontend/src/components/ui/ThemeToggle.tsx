// src/components/ui/ThemeToggle.tsx
import { useTheme } from "next-themes";
import { Sun, Moon, Laptop } from "lucide-react";
import { Button } from "./button"; // shadcn/ui button (or your own)

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();

  return (
    <div className="flex gap-1 p-1 rounded-lg bg-white/5 border border-white/10">
      <Button
        variant="ghost"
        size="sm"
        onClick={() => setTheme("light")}
        className={theme === "light" ? "bg-neon-blue/20 text-neon-blue" : ""}
      >
        <Sun size={16} />
      </Button>
      <Button
        variant="ghost"
        size="sm"
        onClick={() => setTheme("dark")}
        className={theme === "dark" ? "bg-neon-blue/20 text-neon-blue" : ""}
      >
        <Moon size={16} />
      </Button>
      <Button
        variant="ghost"
        size="sm"
        onClick={() => setTheme("system")}
        className={theme === "system" ? "bg-neon-blue/20 text-neon-blue" : ""}
      >
        <Laptop size={16} />
      </Button>
    </div>
  );
}
