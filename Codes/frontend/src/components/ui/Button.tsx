import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "../../lib/utils";
import { Loader2 } from "lucide-react";

const buttonVariants = cva(
  "inline-flex items-center justify-center rounded-xl font-semibold transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-neon-blue/50 disabled:opacity-50 disabled:cursor-not-allowed active:scale-[0.97] select-none",
  {
    variants: {
      variant: {
        default:
          "bg-neon-blue/90 text-white hover:bg-neon-blue shadow-lg shadow-neon-blue/20 hover:shadow-neon-blue/40",
        primary:
          "bg-neon-blue/90 text-white hover:bg-neon-blue shadow-lg shadow-neon-blue/20 hover:shadow-neon-blue/40",
        secondary:
          "bg-white/10 text-slate-300 hover:bg-white/20 border border-white/10",
        ghost:
          "bg-transparent text-slate-400 hover:text-white hover:bg-white/5",
        danger:
          "bg-red-600/80 text-white hover:bg-red-600 shadow-lg shadow-red-500/20",
        success:
          "bg-green-600/80 text-white hover:bg-green-600 shadow-lg shadow-green-500/20",
        outline:
          "border border-neon-blue/50 text-neon-blue hover:bg-neon-blue/10",
      },
      size: {
        sm: "px-3 py-1.5 text-xs gap-1.5",
        md: "px-5 py-2.5 text-sm gap-2",
        lg: "px-6 py-3 text-base gap-2.5",
        icon: "p-2",
      },
    },
    defaultVariants: { variant: "primary", size: "md" },
  },
);

interface ButtonProps
  extends
    React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  loading?: boolean;
  icon?: React.ReactNode;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    { className, variant, size, loading, children, disabled, icon, ...props },
    ref,
  ) => {
    return (
      <button
        className={cn(buttonVariants({ variant, size, className }))}
        disabled={disabled || loading}
        ref={ref}
        {...props}
      >
        {loading && <Loader2 size={16} className="animate-spin shrink-0" />}
        {!loading && icon && <span className="shrink-0">{icon}</span>}
        {children}
      </button>
    );
  },
);
Button.displayName = "Button";
