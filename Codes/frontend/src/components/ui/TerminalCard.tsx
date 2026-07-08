// src/components/ui/TerminalCard.tsx
import { cn } from "../../lib/utils";

interface TerminalCardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "glowing" | "dense";
  noPadding?: boolean;
}

export const TerminalCard = ({
  children,
  className,
  variant = "default",
  noPadding = false,
  ...props
}: TerminalCardProps) => {
  return (
    <div
      className={cn(
        "border rounded-md bg-black/40 backdrop-blur-sm transition-all duration-200",
        variant === "default" && "border-white/10 hover:border-white/20",
        variant === "glowing" &&
          "border-neon-blue/30 shadow-[0_0_15px_rgba(59,130,246,0.1)] hover:shadow-[0_0_20px_rgba(59,130,246,0.2)]",
        variant === "dense" && "border-white/5 bg-black/60",
        !noPadding && "p-4",
        className,
      )}
      {...props}
    >
      {children}
    </div>
  );
};
