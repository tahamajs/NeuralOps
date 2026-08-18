// src/features/experimentExplorer/components/ThreeDScatterPlot.tsx
import Plot from "react-plotly.js";
import { ExperimentRun } from "../types";

interface ThreeDScatterPlotProps {
  experiments: ExperimentRun[];
  xParam: string;
  yParam: string;
  zParam: string;
  colorMetric: string;
}

export function ThreeDScatterPlot({
  experiments = [],
  xParam,
  yParam,
  zParam,
  colorMetric,
}: ThreeDScatterPlotProps) {
  if (!experiments || !Array.isArray(experiments) || experiments.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-slate-500 text-sm py-12">
        No experiment data available for 3D scatter plot.
      </div>
    );
  }

  const colors = experiments.map((exp) => Number(exp?.metrics?.[colorMetric] ?? 0));

  return (
    <Plot
      data={[
        {
          type: "scatter3d",
          mode: "markers",
          x: experiments.map((exp) => Number(exp?.parameters?.[xParam] ?? 0)),
          y: experiments.map((exp) => Number(exp?.parameters?.[yParam] ?? 0)),
          z: experiments.map((exp) => Number(exp?.parameters?.[zParam] ?? 0)),
          marker: {
            size: 5,
            color: colors,
            colorscale: "Viridis",
            showscale: true,
            colorbar: { title: colorMetric || "Metric" },
          },
          text: experiments.map(
            (exp) =>
              `Run: ${exp?.name || exp?.id || "Unnamed"}<br>${xParam}: ${exp?.parameters?.[xParam] ?? "N/A"}<br>${yParam}: ${exp?.parameters?.[yParam] ?? "N/A"}<br>${zParam}: ${exp?.parameters?.[zParam] ?? "N/A"}<br>${colorMetric}: ${exp?.metrics?.[colorMetric] ?? "N/A"}`,
          ),
          hoverinfo: "text",
        },
      ]}
      layout={{
        autosize: true,
        height: 480,
        paper_bgcolor: "rgba(0,0,0,0)",
        scene: {
          xaxis: { title: xParam || "X", color: "#94a3b8" },
          yaxis: { title: yParam || "Y", color: "#94a3b8" },
          zaxis: { title: zParam || "Z", color: "#94a3b8" },
          bgcolor: "rgba(0,0,0,0)",
        },
        margin: { t: 60, r: 60, b: 40, l: 60 },
        font: { color: "#94a3b8" },
      }}
      useResizeHandler
      style={{ width: "100%", height: "100%" }}
    />
  );
}
