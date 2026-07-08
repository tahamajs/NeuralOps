interface OrderBookProps {
  bids: { price: number; quantity: number }[];
  asks: { price: number; quantity: number }[];
}

export function OrderBook({ bids, asks }: OrderBookProps) {
  const maxTotal = Math.max(
    bids.reduce((sum, b) => sum + b.quantity, 0),
    asks.reduce((sum, a) => sum + a.quantity, 0),
  );

  return (
    <div className="glass-terminal p-4 rounded-xl">
      <h3 className="text-sm font-mono text-neon-mint mb-2">
        Order Book Depth
      </h3>
      <div className="grid grid-cols-2 gap-4">
        {/* Bids (BUY) */}
        <div>
          <div className="text-xs text-green-400 mb-1">Bids (BUY)</div>
          {bids.slice(0, 10).map((bid, idx) => (
            <div key={idx} className="flex justify-between text-sm py-0.5">
              <span className="text-white">{bid.price.toFixed(4)}</span>
              <div className="flex-1 mx-2 relative">
                <div
                  className="absolute right-0 top-0 h-full bg-green-500/20"
                  style={{ width: `${(bid.quantity / maxTotal) * 100}%` }}
                />
                <span className="relative z-10 text-slate-300">
                  {bid.quantity}
                </span>
              </div>
            </div>
          ))}
        </div>
        {/* Asks (SELL) */}
        <div>
          <div className="text-xs text-red-400 mb-1">Asks (SELL)</div>
          {asks.slice(0, 10).map((ask, idx) => (
            <div key={idx} className="flex justify-between text-sm py-0.5">
              <span className="text-white">{ask.price.toFixed(4)}</span>
              <div className="flex-1 mx-2 relative">
                <div
                  className="absolute right-0 top-0 h-full bg-red-500/20"
                  style={{ width: `${(ask.quantity / maxTotal) * 100}%` }}
                />
                <span className="relative z-10 text-slate-300">
                  {ask.quantity}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
