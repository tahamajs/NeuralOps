import { useEffect, useState } from "react";
import { useWebSocket } from "@/hooks/useWebSocket";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface GPUMarketData {
  symbol: string;
  price: number;
  change_24h: number;
  volume_24h: number;
  market_cap: number;
  chart_data: { time: number; price: number }[];
}

export function GPUMarketOverview() {
  const [marketData, setMarketData] = useState<GPUMarketData[]>([]);
  const { lastMessage } = useWebSocket("/ws/market/");

  useEffect(() => {
    if (lastMessage) {
      const data = JSON.parse(lastMessage.data);
      setMarketData((prev) =>
        prev.map((item) => (item.symbol === data.symbol ? data : item)),
      );
    }
  }, [lastMessage]);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4">
      {marketData.map((item) => (
        <div key={item.symbol} className="glass-terminal p-4 rounded-xl">
          <div className="flex justify-between items-center">
            <div>
              <span className="text-lg font-bold text-white">
                {item.symbol}
              </span>
              <span className="text-sm text-slate-400 ml-2">
                {item.price.toFixed(4)} USD/hr
              </span>
            </div>
            <span
              className={`text-sm ${item.change_24h >= 0 ? "text-green-400" : "text-red-400"}`}
            >
              {item.change_24h >= 0 ? "+" : ""}
              {item.change_24h.toFixed(2)}%
            </span>
          </div>
          <div className="h-20 mt-2">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={item.chart_data}>
                <Area
                  type="monotone"
                  dataKey="price"
                  stroke="#3b82f6"
                  fill="#3b82f620"
                />
                <Tooltip />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div className="flex justify-between mt-2 text-xs text-slate-500">
            <span>Vol: {item.volume_24h.toLocaleString()} hrs</span>
            <span>MCap: ${item.market_cap.toLocaleString()}</span>
          </div>
        </div>
      ))}
    </div>
  );
}
