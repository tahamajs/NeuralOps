import { useQuery } from "@tanstack/react-query";
import { hpoAPI } from "../services/api";
import Plot from "react-plotly.js";

export function ParallelCoordinates({ sweepId }: { sweepId: string }) {
  const { data, isLoading } = useQuery({
    queryKey: ["parallel", sweepId],
    queryFn: () => hpoAPI.getParallelCoordinatesData(sweepId),
  });
  if (isLoading)
    return (
      <div className="text-slate-500 text-center py-8">
        Loading parallel coordinates...
      </div>
    );
  if (!data)
    return (
      <div className="text-slate-500 text-center py-8">No data available</div>
    );
  return (
    <Plot
      data={[data]}
      layout={{
        autosize: true,
        height: 500,
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "#94a3b8" },
      }}
      useResizeHandler
      style={{ width: "100%" }}
    />
  );
}
