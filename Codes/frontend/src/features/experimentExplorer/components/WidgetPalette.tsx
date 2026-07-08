// src/features/experimentExplorer/components/WidgetPalette.tsx
import { TerminalCard } from "../../../components/ui/TerminalCard";
import {
  Plus,
  BarChart4,
  Grid3X3,
  Table2,
  LineChart,
  PieChart,
} from "lucide-react";

const WIDGET_TYPES = [
  {
    type: "scatter",
    label: "Scatter Plot",
    icon: BarChart4,
    description: "2D parameter relationships",
  },
  {
    type: "parallel_coords",
    label: "Parallel Coordinates",
    icon: Grid3X3,
    description: "Multi‑parameter patterns",
  },
  {
    type: "correlation_matrix",
    label: "Correlation Matrix",
    icon: Table2,
    description: "Parameter impact heatmap",
  },
  {
    type: "metric_history",
    label: "Metric History",
    icon: LineChart,
    description: "Convergence over time",
  },
  {
    type: "parameter_distribution",
    label: "Parameter Distribution",
    icon: PieChart,
    description: "Histogram of values",
  },
];

export function WidgetPalette({
  onAddWidget,
}: {
  onAddWidget: (type: string) => void;
}) {
  return (
    <TerminalCard className="p-4">
      <h3 className="text-sm font-bold text-white mb-3">Add Widget</h3>
      <div className="space-y-2">
        {WIDGET_TYPES.map((w) => (
          <button
            key={w.type}
            onClick={() => onAddWidget(w.type)}
            className="w-full text-left p-2 rounded hover:bg-white/5 flex items-center gap-3"
          >
            <w.icon size={16} className="text-neon-mint" />
            <div>
              <div className="text-sm text-white">{w.label}</div>
              <div className="text-xs text-slate-400">{w.description}</div>
            </div>
          </button>
        ))}
      </div>
    </TerminalCard>
  );
}
