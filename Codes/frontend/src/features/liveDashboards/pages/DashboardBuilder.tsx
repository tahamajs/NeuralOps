// src/features/liveDashboards/pages/DashboardBuilder.tsx
import { useState, useEffect, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import ReactGridLayout from "react-grid-layout";
const GridLayout = ReactGridLayout as any;
import "react-grid-layout/css/styles.css";
import "react-resizable/css/styles.css";
import { liveDashboardsAPI } from "../services/api";
import { useLiveDashboard } from "../hooks/useLiveDashboard";
import { WidgetRenderer } from "../components/WidgetRenderer";
import { WidgetConfigModal } from "../components/WidgetConfigModal";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import {
  Save,
  Settings,
  Play,
  Square,
  RefreshCw,
  Plus,
  Trash2,
  LayoutGrid,
} from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function DashboardBuilder({
  onClose,
}: {
  onClose?: () => void;
}) {
  const { dashboardId } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [editingWidgetId, setEditingWidgetId] = useState<string | null>(null);
  const [refreshInterval, setRefreshInterval] = useState(5);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [layout, setLayout] = useState<any[]>([]);
  const { success, error } = useToast();

  const { data: dashboard, isLoading } = useQuery({
    queryKey: ["dashboard", dashboardId],
    queryFn: () => liveDashboardsAPI.getDashboard(dashboardId!),
    enabled: !!dashboardId,
  });

  const { liveData, isConnected } = useLiveDashboard(
    dashboardId || null,
    dashboard?.widgets || [],
    autoRefresh ? refreshInterval : 0,
  );

  const updateDashboard = useMutation({
    mutationFn: (data: any) =>
      liveDashboardsAPI.updateDashboard(dashboardId!, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["dashboard", dashboardId] });
      success("Dashboard saved");
    },
  });

  const handleLayoutChange = useCallback(
    (newLayout: any[]) => {
      setLayout(newLayout);
      if (dashboard) {
        const updatedWidgets = dashboard.widgets.map((w) => {
          const pos = newLayout.find((l) => l.i === w.id);
          if (pos)
            return {
              ...w,
              position: { x: pos.x, y: pos.y, w: pos.w, h: pos.h },
            };
          return w;
        });
        updateDashboard.mutate({ widgets: updatedWidgets });
      }
    },
    [dashboard, updateDashboard],
  );

  const addWidget = (type: string) => {
    // Implementation would open WidgetLibrary or directly add a default widget
    // For brevity, we'll use a simple addition
    const newId = `widget-${Date.now()}`;
    const newWidget = {
      id: newId,
      type: type as any,
      title: `New ${type}`,
      data_source: { type: "experiment", query: "" },
      position: { x: 0, y: Infinity, w: 4, h: 3 },
    };
    const updatedWidgets = [...(dashboard?.widgets || []), newWidget];
    updateDashboard.mutate({ widgets: updatedWidgets });
  };

  const removeWidget = (widgetId: string) => {
    const updatedWidgets =
      dashboard?.widgets.filter((w) => w.id !== widgetId) || [];
    updateDashboard.mutate({ widgets: updatedWidgets });
  };

  if (isLoading) return <Spinner />;
  if (!dashboard)
    return <div className="p-4 text-red-400">Dashboard not found</div>;

  const gridItems = dashboard.widgets.map((w) => ({
    i: w.id,
    x: w.position.x,
    y: w.position.y,
    w: w.position.w,
    h: w.position.h,
  }));

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <Input
            value={dashboard.name}
            onChange={(e) => updateDashboard.mutate({ name: e.target.value })}
            className="text-xl font-bold bg-transparent border-none px-0"
          />
          <div className="text-xs text-slate-400 mt-1">
            {isConnected ? "● Live" : "○ Reconnecting"}
          </div>
        </div>
        <div className="flex gap-2">
          <div className="flex items-center gap-2 mr-2">
            <span className="text-xs text-slate-400">Refresh:</span>
            <Select
              options={[
                { value: "0", label: "Off" },
                { value: "1", label: "1s" },
                { value: "5", label: "5s" },
                { value: "10", label: "10s" },
                { value: "30", label: "30s" },
              ]}
              value={refreshInterval.toString()}
              onChange={(e) => setRefreshInterval(parseInt(e.target.value))}
              className="w-20"
            />
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`p-1 rounded ${autoRefresh ? "text-neon-mint" : "text-slate-400"}`}
            >
              <Play size={14} />
            </button>
            <button
              onClick={() => setAutoRefresh(false)}
              className="p-1 rounded text-slate-400 hover:text-white"
            >
              <Square size={14} />
            </button>
          </div>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => navigate("/dashboard-widgets")}
          >
            <LayoutGrid size={14} className="mr-1" /> Widget Library
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => window.location.reload()}
          >
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
        <GridLayout
          className="layout"
          layout={gridItems}
          cols={12}
          rowHeight={30}
          width={1200}
          onLayoutChange={handleLayoutChange}
          draggableHandle=".drag-handle"
        >
          {dashboard.widgets.map((w) => (
            <div key={w.id} className="glass-terminal flex flex-col">
              <div className="drag-handle cursor-move p-2 border-b border-white/10 flex justify-between items-center bg-black/30">
                <span className="text-sm font-medium text-white">
                  {w.title}
                </span>
                <div className="flex gap-1">
                  <button
                    onClick={() => setEditingWidgetId(w.id)}
                    className="text-slate-400 hover:text-white"
                  >
                    <Settings size={12} />
                  </button>
                  <button
                    onClick={() => removeWidget(w.id)}
                    className="text-slate-400 hover:text-red-400"
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              </div>
              <div className="flex-1 p-2 overflow-auto">
                <WidgetRenderer widget={w} data={liveData[w.id] || []} />
              </div>
            </div>
          ))}
        </GridLayout>
      </div>
      <WidgetConfigModal
        open={!!editingWidgetId}
        widget={dashboard.widgets.find((w) => w.id === editingWidgetId)}
        onClose={() => setEditingWidgetId(null)}
        onSave={(updatedWidget) => {
          const updatedWidgets = dashboard.widgets.map((w) =>
            w.id === updatedWidget.id ? updatedWidget : w,
          );
          updateDashboard.mutate({ widgets: updatedWidgets });
          setEditingWidgetId(null);
        }}
      />
    </div>
  );
}
