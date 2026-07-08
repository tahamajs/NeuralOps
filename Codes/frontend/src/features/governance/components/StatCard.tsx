import { TerminalCard } from "../../../components/ui/TerminalCard";

export function StatCard({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string | number;
}) {
  return (
    <TerminalCard className="p-3 text-center">
      <div className="flex items-center justify-center gap-1 text-slate-400 mb-1">
        {icon}
      </div>
      <div className="text-2xl font-bold text-white">{value}</div>
      <div className="text-xs text-slate-500">{label}</div>
    </TerminalCard>
  );
}
