// src/features/gpuFinance/components/PositionCard.tsx
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Badge } from "../../../components/ui/Badge";
import { TrendingUp, TrendingDown, AlertCircle } from "lucide-react";

interface Position {
  contract_id: string;
  side: "long" | "short";
  quantity: number;
  entry_price: number;
  mark_price: number;
  unrealized_pnl: number;
  realized_pnl: number;
  liquidation_price?: number;
}

interface Contract {
  id: string;
  underlying_index: string;
  expiry_date: string;
}

export function PositionCard({
  position,
  contract,
}: {
  position: Position;
  contract?: Contract;
}) {
  const pnlAbs = position.unrealized_pnl;
  const pnlPercent =
    ((position.mark_price - position.entry_price) / position.entry_price) *
    100 *
    (position.side === "long" ? 1 : -1);
  const roi =
    ((position.mark_price - position.entry_price) / position.entry_price) * 100;

  const isProfit = pnlAbs >= 0;
  const sideColor = position.side === "long" ? "green" : "red";
  const pnlColor = isProfit ? "text-green-400" : "text-red-400";

  return (
    <TerminalCard className="p-3">
      <div className="flex justify-between items-start">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-bold text-white">
              {contract?.underlying_index || position.contract_id.slice(0, 8)}
            </span>
            <Badge variant={position.side === "long" ? "success" : "danger"}>
              {position.side.toUpperCase()} {position.quantity}
            </Badge>
          </div>
          <div className="text-xs text-slate-400 mt-1">
            Entry: ${position.entry_price.toFixed(4)} · Mark: $
            {position.mark_price.toFixed(4)}
          </div>
        </div>
        <div className="text-right">
          <div className={`text-sm font-bold ${pnlColor}`}>
            {isProfit ? "+" : ""}
            {pnlAbs.toFixed(2)} USD
          </div>
          <div className="text-xs text-slate-400">{pnlPercent.toFixed(2)}%</div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 mt-2 pt-2 border-t border-white/10 text-xs">
        <div>
          <div className="text-slate-400">ROI</div>
          <div className={roi >= 0 ? "text-green-400" : "text-red-400"}>
            {roi.toFixed(2)}%
          </div>
        </div>
        <div>
          <div className="text-slate-400">Realised PnL</div>
          <div
            className={
              position.realized_pnl >= 0 ? "text-green-400" : "text-red-400"
            }
          >
            ${position.realized_pnl.toFixed(2)}
          </div>
        </div>
        <div>
          <div className="text-slate-400">Liq. Price</div>
          <div className="text-white">
            {position.liquidation_price?.toFixed(4) || "N/A"}
          </div>
        </div>
      </div>

      {contract?.expiry_date && new Date(contract.expiry_date) < new Date() && (
        <div className="mt-2 flex items-center gap-1 text-xs text-yellow-400 bg-yellow-500/10 p-1 rounded">
          <AlertCircle size={12} /> Expired on{" "}
          {new Date(contract.expiry_date).toLocaleDateString()}
        </div>
      )}
    </TerminalCard>
  );
}
