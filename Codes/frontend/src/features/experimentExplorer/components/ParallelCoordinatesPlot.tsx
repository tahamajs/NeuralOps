// src/features/experimentExplorer/components/ParallelCoordinatesPlot.tsx
import { useEffect, useRef } from "react";
import Plot from "react-plotly.js";
import { ExperimentRun } from "../types";

export function ParallelCoordinatesPlot({
  experiments = [],
  targetMetric,
}: {
  experiments: ExperimentRun[];
  targetMetric: string;
}) {
  const plotRef = useRef<any>();

  if (!experiments || !Array.isArray(experiments) || experiments.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-slate-500 text-sm py-12">
        No experiment data available.
      </div>
    );
  }

  // Extract all numeric parameter names across experiments
  const paramSet = new Set<string>();
  experiments.forEach((exp) => {
    if (exp?.parameters && typeof exp.parameters === "object") {
      Object.entries(exp.parameters).forEach(([k, v]) => {
        if (typeof v === "number" && !isNaN(v)) paramSet.add(k);
      });
    }
  });
  const numericParams = Array.from(paramSet);

  if (numericParams.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-slate-500 text-sm py-12">
        No numeric parameters found in experiments.
      </div>
    );
  }

  const dimensions = numericParams.map((param) => {
    const vals = experiments.map((exp) => Number(exp?.parameters?.[param] ?? 0));
    const validVals = vals.filter((n) => typeof n === "number" && !isNaN(n));
    const minVal = validVals.length > 0 ? Math.min(...validVals) : 0;
    const maxVal = validVals.length > 0 ? Math.max(...validVals) : 1;
    return {
      label: param,
      values: vals,
      range: [minVal, maxVal === minVal ? minVal + 1 : maxVal],
    };
  });

  const colors = experiments.map((exp) => Number(exp?.metrics?.[targetMetric] ?? 0));
  const validColors = colors.filter((n) => typeof n === "number" && !isNaN(n));
  const cmin = validColors.length > 0 ? Math.min(...validColors) : 0;
  const cmax = validColors.length > 0 ? Math.max(...validColors) : 1;

  return (
    <Plot
      data={[
        {
          type: "parcoords",
          dimensions,
          line: {
            color: colors,
            colorscale: "Viridis",
            showscale: true,
            cmin,
            cmax: cmax === cmin ? cmin + 1 : cmax,
          },
          labelfont: { color: "#94a3b8" },
          tickfont: { color: "#94a3b8" },
        },
      ]}
      layout={{
        autosize: true,
        height: 480,
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        margin: { t: 60, r: 80, b: 40, l: 80 },
        font: { color: "#94a3b8" },
        coloraxis: {
          colorscale: "Viridis",
          colorbar: { title: { text: targetMetric || "metric" } },
        },
      }}
      useResizeHandler
      style={{ width: "100%", height: "100%" }}
    />
  );
}
