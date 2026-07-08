import { useState, useEffect } from "react";

export interface OrderBookData {
  asks: [number, number][];
  bids: [number, number][];
  last_price: number;
}

export function useTradingWebSocket(gpuModel: string) {
  const [orderBook, setOrderBook] = useState<OrderBookData>({
    asks: [
      [1.25, 100],
      [1.24, 250],
      [1.23, 150],
      [1.22, 300],
      [1.21, 500],
    ],
    bids: [
      [1.19, 400],
      [1.18, 200],
      [1.17, 600],
      [1.16, 150],
      [1.15, 800],
    ],
    last_price: 1.20,
  });

  useEffect(() => {
    // Basic simulation of changes in price/asks/bids
    const interval = setInterval(() => {
      setOrderBook((prev) => {
        const delta = (Math.random() - 0.5) * 0.02;
        const last_price = parseFloat((prev.last_price + delta).toFixed(2));
        const newAsks = prev.asks.map(([p, q]) => [
          parseFloat((p + delta).toFixed(2)),
          Math.max(10, q + Math.floor((Math.random() - 0.5) * 50)),
        ]) as [number, number][];
        const newBids = prev.bids.map(([p, q]) => [
          parseFloat((p + delta).toFixed(2)),
          Math.max(10, q + Math.floor((Math.random() - 0.5) * 50)),
        ]) as [number, number][];

        return {
          asks: newAsks.sort((a, b) => a[0] - b[0]),
          bids: newBids.sort((a, b) => b[0] - a[0]),
          last_price,
        };
      });
    }, 3000);

    return () => clearInterval(interval);
  }, [gpuModel]);

  return { orderBook };
}
