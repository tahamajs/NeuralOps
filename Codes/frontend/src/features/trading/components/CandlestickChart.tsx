import React from "react";
import { useQuery } from "@tanstack/react-query";
import Plot from "react-plotly.js";
import api from "../../../api/client";

export default function CandlestickChart({ gpuModel }: { gpuModel: string }) {
  const { data: candles } = useQuery({
    queryKey: ["candles", gpuModel],
    queryFn: () =>
      api
        .get(`/trading/orders/candles/?gpu_model=${gpuModel}&limit=50`)
        .then((r) => r.data),
    refetchInterval: 10000,
  });

  if (!candles?.length)
    return <div className="glass p-4 h-64">Loading chart...</div>;

  const trace = {
    x: candles.map((c: any) => c.interval_start),
    open: candles.map((c: any) => c.open_price),
    high: candles.map((c: any) => c.high_price),
    low: candles.map((c: any) => c.low_price),
    close: candles.map((c: any) => c.close_price),
    type: "candlestick" as const,
    name: gpuModel,
    increasing: { line: { color: "#22c55e" } },
    decreasing: { line: { color: "#ef4444" } },
  };

  return (
    <div className="glass p-4">
      <h2 className="text-lg font-bold mb-2">{gpuModel} Price Chart</h2>
      <Plot
        data={[trace]}
        layout={{
          autosize: true,
          margin: { t: 10, r: 20, b: 30, l: 50 },
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
          font: { color: "#94a3b8" },
          xaxis: { gridcolor: "#1e293b", type: "date" },
          yaxis: { gridcolor: "#1e293b", title: "Price (USD/hour)" },
        }}
        useResizeHandler
        style={{ width: "100%", height: "400px" }}
      />
    </div>
  );
}
