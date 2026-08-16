// src/features/marketplace/components/OrderBook.tsx
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { marketplaceAPI } from "../services/api";
import { useOrderBookWebSocket } from "./useOrderBookWebSocket";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Spinner } from "../../../components/ui/Spinner";
import { TrendingUp, TrendingDown, Activity } from "lucide-react";

interface OrderBookProps {
  listingId: string;
}

interface OrderBookLevel {
  price: number;
  quantity: number;
  total: number;
}

export function OrderBook({ listingId }: OrderBookProps) {
  const [view, setView] = useState<"table" | "depth">("table");
  const {
    data: depth,
    isLoading,
    refetch,
  } = useQuery({
    queryKey: ["order-book-depth", listingId],
    queryFn: () => marketplaceAPI.getOrderBookDepth(listingId),
    refetchInterval: 5000,
  });
  const { liveDepth, isConnected } = useOrderBookWebSocket(listingId);
  const currentDepth = liveDepth || depth;

  if (isLoading) return <Spinner size="sm" />;

  const maxTotal = Math.max(
    ...(currentDepth?.bids.map((b: OrderBookLevel) => b.total) || []),
    ...(currentDepth?.asks.map((a: OrderBookLevel) => a.total) || []),
  );

  return (
    <TerminalCard className="p-4">
      <div className="flex justify-between items-center mb-3">
        <div className="flex gap-2">
          <button
            onClick={() => setView("table")}
            className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
              view === "table"
                ? "bg-neon-blue text-white"
                : "bg-white/5 text-slate-400 hover:bg-white/10"
            }`}
          >
            Order Book
          </button>
          <button
            onClick={() => setView("depth")}
            className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
              view === "depth"
                ? "bg-neon-blue text-white"
                : "bg-white/5 text-slate-400 hover:bg-white/10"
            }`}
          >
            Depth Chart
          </button>
        </div>
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${isConnected ? "bg-green-400" : "bg-red-400"}`}
          />
          <span className="text-xs text-slate-400">
            {isConnected ? "Live" : "Polling"}
          </span>
        </div>
      </div>

      {view === "table" ? (
        <div className="grid grid-cols-2 gap-4">
          {/* Bids */}
          <div>
            <div className="text-xs font-bold text-green-400 uppercase tracking-wider mb-2">
              Bids (Buy)
            </div>
            <div className="space-y-1 max-h-96 overflow-y-auto">
              {currentDepth?.bids
                .slice(0, 15)
                .map((bid: OrderBookLevel, idx: number) => (
                  <div key={idx} className="flex justify-between text-sm">
                    <span className="font-mono text-white">
                      {bid.price.toFixed(4)}
                    </span>
                    <span className="text-slate-400">
                      {bid.quantity.toFixed(2)}
                    </span>
                    <span className="text-slate-500">
                      {bid.total.toFixed(0)}
                    </span>
                  </div>
                ))}
            </div>
          </div>
          {/* Asks */}
          <div>
            <div className="text-xs font-bold text-red-400 uppercase tracking-wider mb-2">
              Asks (Sell)
            </div>
            <div className="space-y-1 max-h-96 overflow-y-auto">
              {currentDepth?.asks
                .slice(0, 15)
                .map((ask: OrderBookLevel, idx: number) => (
                  <div key={idx} className="flex justify-between text-sm">
                    <span className="font-mono text-white">
                      {ask.price.toFixed(4)}
                    </span>
                    <span className="text-slate-400">
                      {ask.quantity.toFixed(2)}
                    </span>
                    <span className="text-slate-500">
                      {ask.total.toFixed(0)}
                    </span>
                  </div>
                ))}
            </div>
          </div>
        </div>
      ) : (
        <div className="relative h-64">
          <div className="absolute inset-0 flex items-end">
            {currentDepth?.bids.map((bid: OrderBookLevel, idx: number) => (
              <div
                key={`bid-${idx}`}
                className="bg-green-500/30 border-t border-green-500/50"
                style={{
                  height: `${(bid.total / maxTotal) * 100}%`,
                  width: `${100 / (currentDepth.bids.length + currentDepth.asks.length)}%`,
                  left: `${idx * (100 / (currentDepth.bids.length + currentDepth.asks.length))}%`,
                }}
              />
            ))}
            {currentDepth?.asks.map((ask: OrderBookLevel, idx: number) => (
              <div
                key={`ask-${idx}`}
                className="bg-red-500/30 border-t border-red-500/50"
                style={{
                  height: `${(ask.total / maxTotal) * 100}%`,
                  width: `${100 / (currentDepth.bids.length + currentDepth.asks.length)}%`,
                  left: `${(currentDepth.bids.length + idx) * (100 / (currentDepth.bids.length + currentDepth.asks.length))}%`,
                }}
              />
            ))}
          </div>
          <div className="absolute bottom-0 left-0 right-0 flex justify-between text-xs text-slate-500 px-2">
            <span>Price ↓</span>
            <span>Volume</span>
          </div>
        </div>
      )}

      <div className="mt-3 flex justify-between text-xs text-slate-500 border-t border-white/10 pt-2">
        <div>Spread: {currentDepth?.spread?.toFixed(4)}</div>
        <div>Last Price: {currentDepth?.last_price?.toFixed(4)}</div>
      </div>
    </TerminalCard>
  );
}
