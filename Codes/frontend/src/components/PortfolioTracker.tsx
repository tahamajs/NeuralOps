import { useQuery } from "@tanstack/react-query";
import { marketplaceApi } from "../api/marketplace";
import { TerminalCard } from "./ui/TerminalCard";
import { TrendingUp, TrendingDown, Wallet } from "lucide-react";

export function PortfolioTracker() {
  const {
    data: portfolio,
    isLoading,
    refetch,
  } = useQuery({
    queryKey: ["portfolio"],
    queryFn: () => marketplaceApi.getPortfolio(),
    refetchInterval: 10000,
  });

  if (isLoading)
    return <div className="glass-terminal p-4">Loading portfolio...</div>;
  if (!portfolio) return null;

  const totalPnL = portfolio.unrealized_pnl_display;
  const isPositive = totalPnL >= 0;

  return (
    <div className="glass-terminal p-4 rounded-xl">
      <div className="flex items-center gap-2 mb-3">
        <Wallet size={18} className="text-neon-mint" />
        <h3 className="text-sm font-mono text-neon-mint">My Portfolio</h3>
      </div>
      <div className="grid grid-cols-2 gap-4 text-center">
        <div>
          <div className="text-2xl font-bold text-white">
            ${portfolio.total_value_usd_display.toFixed(2)}
          </div>
          <div className="text-xs text-slate-400">Total Value</div>
        </div>
        <div>
          <div
            className={`text-2xl font-bold ${isPositive ? "text-green-400" : "text-red-400"}`}
          >
            {isPositive ? "+" : ""}
            {totalPnL.toFixed(2)}%
          </div>
          <div className="text-xs text-slate-400">Unrealized P&L</div>
        </div>
      </div>
      <div className="mt-3 text-xs text-slate-400 flex justify-between">
        <span>Cash Balance: ${portfolio.balance_usd.toFixed(2)}</span>
        <span>Invested: ${portfolio.total_invested_usd.toFixed(2)}</span>
      </div>
      {portfolio.holdings.length > 0 && (
        <div className="mt-3 border-t border-white/10 pt-2">
          <div className="text-xs font-bold text-slate-400 mb-1">Holdings</div>
          {portfolio.holdings.map((holding) => (
            <div
              key={holding.listing_name}
              className="flex justify-between text-sm py-1"
            >
              <span className="text-white">{holding.listing_name}</span>
              <span>{holding.quantity} hrs</span>
              <span
                className={holding.pnl >= 0 ? "text-green-400" : "text-red-400"}
              >
                {holding.pnl >= 0 ? "+" : ""}
                {holding.pnl.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
