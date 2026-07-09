import React from "react";
import { Card } from "../../../components/ui/Card";
import { Activity, Server, Cpu } from "lucide-react";

export function DashboardCard({ title, value, unit, icon: Icon = Activity, trend, status = "normal" }: any) {
  const statusColors = {
    normal: "text-neon-blue",
    warning: "text-amber-400",
    critical: "text-red-400",
    success: "text-green-400",
  };

  return (
    <Card className="glass border-white/5 hover:border-white/10 transition-all p-5 relative overflow-hidden group">
      <div className={`absolute inset-0 opacity-0 group-hover:opacity-5 transition-opacity bg-gradient-to-br from-current to-transparent ${statusColors[status as keyof typeof statusColors]}`}></div>
      
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-slate-400 uppercase tracking-wider">{title || "Metric"}</h3>
        <div className={`p-2 rounded-lg bg-white/5 ${statusColors[status as keyof typeof statusColors]}`}>
          <Icon size={18} />
        </div>
      </div>
      
      <div className="flex items-baseline gap-2">
        <span className="text-3xl font-display font-bold text-white tracking-tight">{value || "0"}</span>
        {unit && <span className="text-sm font-medium text-slate-400">{unit}</span>}
      </div>
      
      {trend && (
        <div className={`mt-3 text-xs font-medium ${trend > 0 ? 'text-green-400' : 'text-red-400'} flex items-center gap-1`}>
          {trend > 0 ? '↑' : '↓'} {Math.abs(trend)}% from last hour
        </div>
      )}
    </Card>
  );
}
