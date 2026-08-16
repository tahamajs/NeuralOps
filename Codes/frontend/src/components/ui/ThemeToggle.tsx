// src/components/ui/ThemeToggle.tsx
import { useState, useEffect } from "react";
import { useTheme } from "next-themes";
import { Sun, Moon, Laptop } from "lucide-react";
import { Button } from "./Button";

interface ThemeToggleProps {
  variant?: "segmented" | "button";
  className?: string;
}

export function ThemeToggle({ variant = "button", className = "" }: ThemeToggleProps) {
  const { theme, setTheme, resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <div className={`w-8 h-8 rounded-lg bg-white/5 animate-pulse ${className}`} />
    );
  }

  if (variant === "button") {
    const isDark = resolvedTheme === "dark";
    return (
      <button
        onClick={() => setTheme(isDark ? "light" : "dark")}
        className={`p-2 rounded-xl text-slate-400 hover:text-white hover:bg-white/5 transition-colors ${className}`}
        aria-label="Toggle dark/light theme"
        title={`Switch to ${isDark ? "light" : "dark"} mode`}
      >
        {isDark ? <Sun size={18} /> : <Moon size={18} />}
      </button>
    );
  }

  return (
    <div
      className={`flex items-center gap-1 p-1 rounded-xl bg-white/5 border border-white/10 ${className}`}
    >
      <Button
        variant="ghost"
        size="sm"
        onClick={() => setTheme("light")}
        className={`px-2 py-1 h-7 rounded-lg ${
          theme === "light"
            ? "bg-neon-blue/20 text-neon-blue font-bold shadow-sm"
            : "text-slate-400 hover:text-white"
        }`}
        title="Light Mode"
      >
        <Sun size={14} />
      </Button>
      <Button
        variant="ghost"
        size="sm"
        onClick={() => setTheme("dark")}
        className={`px-2 py-1 h-7 rounded-lg ${
          theme === "dark"
            ? "bg-neon-blue/20 text-neon-blue font-bold shadow-sm"
            : "text-slate-400 hover:text-white"
        }`}
        title="Dark Mode"
      >
        <Moon size={14} />
      </Button>
      <Button
        variant="ghost"
        size="sm"
        onClick={() => setTheme("system")}
        className={`px-2 py-1 h-7 rounded-lg ${
          theme === "system"
            ? "bg-neon-blue/20 text-neon-blue font-bold shadow-sm"
            : "text-slate-400 hover:text-white"
        }`}
        title="System Preference"
      >
        <Laptop size={14} />
      </Button>
    </div>
  );
}

export default ThemeToggle;
