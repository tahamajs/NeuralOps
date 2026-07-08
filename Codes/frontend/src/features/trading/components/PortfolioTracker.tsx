import React from "react";
import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { PortfolioItem } from "../types";

export default function PortfolioTracker() {
  const { data: portfolio } = useQuery({
    queryKey: ["portfolio"],
    queryFn: () => api.get("/trading/portfolio/").then((r) => r.data),
    refetchInterval: 10000,
  });

  return (
    <div className="glass p-4">
      <h2 className="text-lg font-bold mb-2">My Portfolio</h2>
      <div className="space-y-2">
        {portfolio?.map((item: PortfolioItem) => (
          <div key={item.gpu_model} className="flex justify-between">
            <span>{item.gpu_model}</span>
            <span
              className={item.quantity > 0 ? "text-green-400" : "text-red-400"}
            >
              {item.quantity} hrs
            </span>
            <span>${item.market_value?.toFixed(2) || "—"}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
