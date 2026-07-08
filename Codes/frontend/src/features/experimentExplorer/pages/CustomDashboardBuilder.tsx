// src/features/experimentExplorer/pages/CustomDashboardBuilder.tsx
import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import GridLayout from "react-grid-layout";
import "react-grid-layout/css/styles.css";
import "react-resizable/css/styles.css";
import { experimentExplorerAPI } from "../services/api";
import { WidgetPalette } from "../components/WidgetPalette";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Modal } from "../../../components/ui/Modal";
import { Spinner } from "../../../components/ui/Spinner";
import {
  LayoutDashboard,
  Save,
  FolderOpen,
  Plus,
  Trash2,
  Maximize2,
} from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

const WIDGET_COMPONENTS: Record<string, React.ComponentType<any>> = {
  scatter: () => (
    <div className="bg-white/5 rounded p-2 h-full flex items-center justify-center text-slate-400">
      Scatter Plot
    </div>
  ),
  parallel_coords: () => (
    <div className="bg-white/5 rounded p-2 h-full flex items-center justify-center text-slate-400">
      Parallel Coordinates
    </div>
  ),
  correlation_matrix: () => (
    <div className="bg-white/5 rounded p-2 h-full flex items-center justify-center text-slate-400">
      Correlation Matrix
    </div>
  ),
  metric_history: () => (
    <div className="bg-white/5 rounded p-2 h-full flex items-center justify-center text-slate-400">
      Metric History
    </div>
  ),
  parameter_distribution: () => (
    <div className="bg-white/5 rounded p-2 h-full flex items-center justify-center text-slate-400">
      Parameter Distribution
    </div>
  ),
};

export default function CustomDashboardBuilder({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [layouts, setLayouts] = useState<any[]>([]);
  const [selectedLayoutId, setSelectedLayoutId] = useState<string | null>(null);
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [layoutName, setLayoutName] = useState("");
  const [currentWidgets, setCurrentWidgets] = useState<any[]>([]);
  const [gridItems, setGridItems] = useState<
    { i: string; x: number; y: number; w: number; h: number }[]
  >([]);
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: dashboards, isLoading } = useQuery({
    queryKey: ["dashboard-layouts"],
    queryFn: () => experimentExplorerAPI.getDashboardLayouts(),
  });

  const saveLayout = useMutation({
    mutationFn: () =>
      experimentExplorerAPI.saveDashboardLayout({
        name: layoutName,
        widgets: currentWidgets,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["dashboard-layouts"] });
      setShowSaveModal(false);
      setLayoutName("");
      success("Dashboard saved");
    },
  });

  const loadLayout = (layout: any) => {
    setSelectedLayoutId(layout.id);
    setCurrentWidgets(layout.widgets);
    setGridItems(
      layout.widgets.map((w) => ({ i: w.id, x: w.x, y: w.y, w: w.w, h: w.h })),
    );
  };

  const addWidget = (type: string) => {
    const newId = `widget-${Date.now()}`;
    const newWidget: any = {
      id: newId,
      type: type as any,
      title: type.replace("_", " "),
      config: {},
      x: 0,
      y: Infinity,
      w: 4,
      h: 3,
    };
    setCurrentWidgets((prev) => [...prev, newWidget]);
    setGridItems((prev) => [
      ...prev,
      { i: newId, x: 0, y: Infinity, w: 4, h: 3 },
    ]);
  };

  const removeWidget = (id: string) => {
    setCurrentWidgets((prev) => prev.filter((w) => w.id !== id));
    setGridItems((prev) => prev.filter((i) => i.i !== id));
  };

  const onLayoutChange = (layout: any[]) => {
    setGridItems(
      layout.map((item) => ({
        i: item.i,
        x: item.x,
        y: item.y,
        w: item.w,
        h: item.h,
      })),
    );
    setCurrentWidgets((prev) =>
      prev.map((w) => {
        const updated = layout.find((l) => l.i === w.id);
        if (updated)
          return {
            ...w,
            x: updated.x,
            y: updated.y,
            w: updated.w,
            h: updated.h,
          };
        return w;
      }),
    );
  };

  if (isLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <LayoutDashboard size={20} className="text-neon-mint" /> Custom
            Dashboard Builder
          </h1>
          <p className="text-xs text-slate-400">
            Drag & drop widgets to create your own dashboards
          </p>
        </div>
        <div className="flex gap-2">
          <Select
            options={
              dashboards?.map((d) => ({ value: d.id, label: d.name })) || []
            }
            value={selectedLayoutId || ""}
            onChange={(e) =>
              loadLayout(dashboards?.find((d) => d.id === e.target.value)!)
            }
            placeholder="Load saved..."
            className="w-40"
          />
          <Button size="sm" onClick={() => setShowSaveModal(true)}>
            <Save size={14} className="mr-1" /> Save
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-hidden p-4">
        <div className="grid lg:grid-cols-4 gap-4 h-full">
          <div className="lg:col-span-1 overflow-y-auto">
            <WidgetPalette onAddWidget={addWidget} />
            <div className="mt-4 space-y-2">
              {currentWidgets.map((w) => (
                <div
                  key={w.id}
                  className="flex justify-between items-center p-2 rounded bg-white/5"
                >
                  <span className="text-sm text-white">{w.title}</span>
                  <button onClick={() => removeWidget(w.id)}>
                    <Trash2 size={14} className="text-red-400" />
                  </button>
                </div>
              ))}
            </div>
          </div>
          <div className="lg:col-span-3 bg-black/20 rounded-lg p-2 overflow-auto h-full">
            <GridLayout
              className="layout"
              layout={gridItems}
              cols={12}
              rowHeight={30}
              width={800}
              onLayoutChange={onLayoutChange}
              draggableHandle=".drag-handle"
            >
              {currentWidgets.map((w) => {
                const Component = WIDGET_COMPONENTS[w.type];
                return (
                  <div key={w.id} className="glass-terminal">
                    <div className="drag-handle cursor-move p-2 border-b border-white/10 flex justify-between items-center">
                      <span className="text-sm font-medium text-white">
                        {w.title}
                      </span>
                      <button
                        onClick={() => removeWidget(w.id)}
                        className="text-slate-400 hover:text-red-400"
                      >
                        <Maximize2 size={12} />
                      </button>
                    </div>
                    <div className="p-2 h-[calc(100%-40px)] overflow-auto">
                      <Component />
                    </div>
                  </div>
                );
              })}
            </GridLayout>
          </div>
        </div>
      </div>
      <Modal
        open={showSaveModal}
        onClose={() => setShowSaveModal(false)}
        title="Save Dashboard"
      >
        <Input
          label="Dashboard Name"
          value={layoutName}
          onChange={(e) => setLayoutName(e.target.value)}
        />
        <Button
          className="mt-4 w-full"
          onClick={() => saveLayout.mutate()}
          disabled={!layoutName.trim()}
        >
          Save
        </Button>
      </Modal>
    </div>
  );
}
