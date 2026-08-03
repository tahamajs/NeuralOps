import React from "react";
import { Card } from "../../../components/ui/Card";
import { ArrowDownRight, ArrowUpRight } from "lucide-react";

export function OrderBook() {
  const bids = [
    { price: 2.45, amount: 100, total: 245.0 },
    { price: 2.44, amount: 250, total: 610.0 },
    { price: 2.41, amount: 50, total: 120.5 },
    { price: 2.4, amount: 500, total: 1200.0 },
    { price: 2.38, amount: 120, total: 285.6 },
  ];

  const asks = [
    { price: 2.55, amount: 150, total: 382.5 },
    { price: 2.52, amount: 200, total: 504.0 },
    { price: 2.5, amount: 80, total: 200.0 },
    { price: 2.48, amount: 300, total: 744.0 },
    { price: 2.47, amount: 50, total: 123.5 },
  ];

  return (
    <Card className="glass border-white/5 p-4 flex flex-col h-full">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-white font-semibold flex items-center gap-2">
          Order Book{" "}
          <span className="text-xs font-normal text-slate-400 bg-white/5 px-2 py-1 rounded">
            A100-80GB
          </span>
        </h3>
      </div>

      <div className="grid grid-cols-3 text-xs font-medium text-slate-400 mb-2 border-b border-white/10 pb-2 px-2">
        <div>Price ($/hr)</div>
        <div className="text-right">Amount (GPUs)</div>
        <div className="text-right">Total ($)</div>
      </div>

      <div className="flex-1 overflow-y-auto space-y-4">
        <div className="space-y-1">
          {asks.map((ask, idx) => (
            <div
              key={idx}
              className="grid grid-cols-3 text-sm hover:bg-white/5 px-2 py-1 rounded cursor-pointer transition-colors relative group"
            >
              <div className="absolute inset-0 bg-red-500/10 rounded w-[20%] opacity-20 right-0 left-auto"></div>
              <div className="text-red-400 font-medium z-10 flex items-center gap-1">
                <ArrowDownRight
                  size={12}
                  className="opacity-0 group-hover:opacity-100 transition-opacity"
                />
                {ask.price.toFixed(2)}
              </div>
              <div className="text-white text-right z-10">{ask.amount}</div>
              <div className="text-slate-400 text-right z-10">
                {ask.total.toFixed(2)}
              </div>
            </div>
          ))}
        </div>

        <div className="flex items-center justify-between border-y border-white/10 py-2 px-2 my-2 bg-black/20">
          <span className="text-lg font-bold text-white">2.46</span>
          <span className="text-xs text-slate-400">Spread: $0.01</span>
        </div>

        <div className="space-y-1">
          {bids.map((bid, idx) => (
            <div
              key={idx}
              className="grid grid-cols-3 text-sm hover:bg-white/5 px-2 py-1 rounded cursor-pointer transition-colors relative group"
            >
              <div className="absolute inset-0 bg-green-500/10 rounded w-[40%] opacity-20 right-0 left-auto"></div>
              <div className="text-green-400 font-medium z-10 flex items-center gap-1">
                <ArrowUpRight
                  size={12}
                  className="opacity-0 group-hover:opacity-100 transition-opacity"
                />
                {bid.price.toFixed(2)}
              </div>
              <div className="text-white text-right z-10">{bid.amount}</div>
              <div className="text-slate-400 text-right z-10">
                {bid.total.toFixed(2)}
              </div>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}
