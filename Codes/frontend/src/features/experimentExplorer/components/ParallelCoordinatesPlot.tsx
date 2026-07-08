// src/features/experimentExplorer/components/ParallelCoordinatesPlot.tsx
import { useEffect, useRef } from "react";
import Plot from "react-plotly.js";
import { ExperimentRun } from "../types";

export function ParallelCoordinatesPlot({
  experiments,
  targetMetric,
}: {
  experiments: ExperimentRun[];
  targetMetric: string;
}) {
  const plotRef = useRef<any>();

  // Build dimensions array
  const numericParams = Object.keys(experiments[0]?.parameters || {}).filter(
    (key) => typeof experiments[0]?.parameters[key] === "number",
  );
  const dimensions = numericParams.map((param) => ({
    label: param,
    values: experiments.map((exp) => exp.parameters[param] as number),
    range: [
      Math.min(...experiments.map((exp) => exp.parameters[param] as number)),
      Math.max(...experiments.map((exp) => exp.parameters[param] as number)),
    ],
  }));

  const colors = experiments.map((exp) => exp.metrics[targetMetric]);

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
            cmin: Math.min(...colors),
            cmax: Math.max(...colors),
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
          colorbar: { title: { text: targetMetric } },
        },
      }}
      useResizeHandler
      style={{ width: "100%", height: "100%" }}
    />
  );
}
