// src/features/healthMonitor/pages/HealthDashboard.tsx
import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import ReactGridLayout from "react-grid-layout";
const GridLayout = ReactGridLayout as any;
import "react-grid-layout/css/styles.css";
import "react-resizable/css/styles.css";
import { healthMonitorAPI } from "../services/api";
import { useHealthWebSocket } from "../hooks/useHealthWebSocket";
import { DriftChart } from "../components/DriftChart";
import { FeatureHealthCard } from "../components/FeatureHealthCard";
import { HealthMetricGauge } from "../components/HealthMetricGauge";
import { SystemResourceWidget } from "../components/SystemResourceWidget";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import {
  LayoutDashboard,
  RefreshCw,
  Activity,
  Cpu,
  HardDrive,
  Zap,
  Database,
} from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

// Default widget layout
const defaultLayout = [
  { i: "system", x: 0, y: 0, w: 4, h: 4 },
  { i: "drift", x: 4, y: 0, w: 8, h: 4 },
  { i: "performance", x: 0, y: 4, w: 6, h: 4 },
  { i: "services", x: 6, y: 4, w: 6, h: 4 },
];

export default function HealthDashboard({ onClose }: { onClose?: () => void }) {
  const [layout, setLayout] = useState(defaultLayout);
  const { success, error } = useToast();

  const {
    data: systemHealth,
    isLoading: healthLoading,
    refetch: refetchHealth,
  } = useQuery({
    queryKey: ["system-health"],
    queryFn: () => healthMonitorAPI.getSystemHealth(),
    refetchInterval: 30000,
  });

  const { data: driftMonitors, isLoading: driftLoading } = useQuery({
    queryKey: ["drift-monitors"],
    queryFn: () => healthMonitorAPI.getDriftMonitors(),
  });

  const { data: performanceMetrics } = useQuery({
    queryKey: ["performance-metrics"],
    queryFn: () => healthMonitorAPI.getPerformanceMetrics({ hours: 24 }),
    refetchInterval: 60000,
  });

  const { alerts, isConnected } = useHealthWebSocket();

  const activeCriticalAlerts =
    alerts?.filter((a) => a.severity === "critical" && a.status === "active")
      .length || 0;

  if (healthLoading || driftLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Activity size={20} className="text-neon-mint" /> Health Monitor
          </h1>
          <p className="text-xs text-slate-400">
            Real‑time system and model health tracking
          </p>
        </div>
        <div className="flex gap-2">
          <div
            className={`px-2 py-1 rounded text-xs ${isConnected ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"}`}
          >
            {isConnected ? "● Live" : "○ Offline"}
          </div>
          <Button size="sm" variant="ghost" onClick={() => refetchHealth()}>
            <RefreshCw size={14} />
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4">
        {activeCriticalAlerts > 0 && (
          <div className="mb-4 p-3 bg-red-500/20 border border-red-500/30 rounded text-red-400 text-sm">
            ⚠️ {activeCriticalAlerts} critical alert(s) require attention
          </div>
        )}
        <GridLayout
          className="layout"
          layout={layout}
          cols={12}
          rowHeight={30}
          width={1200}
          onLayoutChange={setLayout}
          isDraggable
          isResizable
        >
          <div key="system" className="glass-terminal">
            <div className="drag-handle cursor-move p-2 border-b border-white/10">
              <span className="text-sm font-bold text-white">
                System Resources
              </span>
            </div>
            <div className="p-2">
              <SystemResourceWidget resources={systemHealth?.resources} />
            </div>
          </div>
          <div key="drift" className="glass-terminal">
            <div className="drag-handle cursor-move p-2 border-b border-white/10">
              <span className="text-sm font-bold text-white">
                Drift Detection
              </span>
            </div>
            <div className="p-2 overflow-auto">
              {(Array.isArray(driftMonitors)
                ? driftMonitors
                : ((driftMonitors as any)?.results ?? [])
              )
                .slice(0, 3)
                .map((m: any) => (
                  <FeatureHealthCard key={m.id} monitor={m} />
                ))}
            </div>
          </div>
          <div key="performance" className="glass-terminal">
            <div className="drag-handle cursor-move p-2 border-b border-white/10">
              <span className="text-sm font-bold text-white">
                Performance Metrics
              </span>
            </div>
            <div className="p-2 grid grid-cols-2 gap-2">
              {(Array.isArray(performanceMetrics)
                ? performanceMetrics
                : ((performanceMetrics as any)?.results ?? [])
              )
                .slice(-4)
                .map((m: any) => (
                  <HealthMetricGauge key={m.metric_type} metric={m} />
                ))}
            </div>
          </div>
          <div key="services" className="glass-terminal">
            <div className="drag-handle cursor-move p-2 border-b border-white/10">
              <span className="text-sm font-bold text-white">
                Service Health
              </span>
            </div>
            <div className="p-2 space-y-2">
              {systemHealth?.services &&
                Object.entries(systemHealth.services).map(([name, s]) => (
                  <div key={name} className="flex justify-between items-center">
                    <span className="capitalize">{name}</span>
                    <Badge
                      variant={
                        (s as any).status === "healthy" ? "success" : "failed"
                      }
                    >
                      {(s as any).status}
                    </Badge>
                  </div>
                ))}
            </div>
          </div>
        </GridLayout>
      </div>
    </div>
  );
}
