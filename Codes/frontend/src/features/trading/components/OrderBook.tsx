import React from 'react';
import { useTradingWebSocket } from '../../../hooks/useTradingWebSocket';

export default function OrderBook({ gpuModel }: { gpuModel: string }) {
  const { orderBook } = useTradingWebSocket(gpuModel);

  if (!orderBook) return <div className="glass p-4">Loading order book...</div>;

  return (
    <div className="glass p-4">
      <h2 className="text-lg font-bold mb-2">Order Book</h2>
      <div className="text-sm">
        <h3 className="text-red-400 font-semibold">Asks</h3>
        {orderBook.asks.slice(0, 5).map(([price, qty], i) => (
          <div key={i} className="flex justify-between">
            <span className="text-red-400">${price}</span>
            <span>{qty}</span>
          </div>
        ))}
        <div className="my-2 border-t border-slate-700 pt-2">
          <span className="text-xl font-bold">${orderBook.last_price || '—'}</span>
        </div>
        <h3 className="text-green-400 font-semibold">Bids</h3>
        {orderBook.bids.slice(0, 5).map(([price, qty], i) => (
          <div key={i} className="flex justify-between">
            <span className="text-green-400">${price}</span>
            <span>{qty}</span>
          </div>
        ))}
      </div>
    </div>
  );
}