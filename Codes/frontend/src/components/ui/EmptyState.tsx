import { cn } from "../../lib/utils";

interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
}

export const EmptyState = ({
  icon,
  title,
  description,
  action,
  className,
}: EmptyStateProps) => (
  <div
    className={cn(
      "flex flex-col items-center justify-center py-16 text-center",
      className,
    )}
  >
    {icon && <div className="mb-4 text-slate-600">{icon}</div>}
    <h3 className="text-lg font-bold text-slate-300 mb-2">{title}</h3>
    {description && (
      <p className="text-sm text-slate-500 max-w-xs">{description}</p>
    )}
    {action && <div className="mt-4">{action}</div>}
  </div>
);
