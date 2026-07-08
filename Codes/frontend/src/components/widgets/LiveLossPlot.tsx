import Plot from "react-plotly.js";
import { TrendingDown } from "lucide-react";

export const LiveLossPlot = ({
  metrics,
}: {
  metrics: { loss: number[]; accuracy: number[] };
}) => (
  <div>
    {metrics.loss.length === 0 ? (
      <div className="flex items-center justify-center h-48 text-slate-500 text-xs">
        <TrendingDown size={20} className="mr-2 opacity-30" />
        No training data yet
      </div>
    ) : (
      <Plot
        data={[
          {
            x: metrics.loss.map((_, i) => i + 1),
            y: metrics.loss,
            type: "scatter",
            mode: "lines+markers",
            name: "Loss",
            line: { color: "#f97316", width: 2 },
            marker: { size: 3 },
          },
        ]}
        layout={{
          autosize: true,
          height: 300,
          margin: { t: 10, r: 10, b: 30, l: 40 },
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
          font: { color: "#94a3b8" },
          showlegend: true,
          legend: { x: 1, y: 1, xanchor: "right" },
        }}
        useResizeHandler
        style={{ width: "100%" }}
      />
    )}
  </div>
);
