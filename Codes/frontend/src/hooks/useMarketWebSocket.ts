import { useState, useEffect } from "react";

export interface LiveMarketPrice {
  price_per_hour: number;
  change_percent: number;
  bid: number;
  ask: number;
  volume_24h: number;
  high_24h: number;
  low_24h: number;
}

const DEFAULT_PRICES: Record<string, LiveMarketPrice> = {
  "listing-h100-8x": {
    price_per_hour: 2.15,
    change_percent: 5.24,
    bid: 2.14,
    ask: 2.16,
    volume_24h: 184200,
    high_24h: 2.22,
    low_24h: 2.05,
  },
  "listing-a100-4x": {
    price_per_hour: 1.25,
    change_percent: -1.82,
    bid: 1.24,
    ask: 1.26,
    volume_24h: 96400,
    high_24h: 1.32,
    low_24h: 1.20,
  },
  "listing-gh200": {
    price_per_hour: 1.95,
    change_percent: 8.10,
    bid: 1.93,
    ask: 1.96,
    volume_24h: 42100,
    high_24h: 2.05,
    low_24h: 1.80,
  },
  "listing-rtx4090": {
    price_per_hour: 0.42,
    change_percent: 1.20,
    bid: 0.41,
    ask: 0.43,
    volume_24h: 310500,
    high_24h: 0.46,
    low_24h: 0.39,
  },
};

export function useMarketWebSocket() {
  const [prices, setPrices] = useState<Record<string, LiveMarketPrice>>(DEFAULT_PRICES);

  useEffect(() => {
    // Dynamic price tick generator
    const interval = setInterval(() => {
      setPrices((prev) => {
        const next = { ...prev };
        const keys = Object.keys(next);
        if (keys.length > 0) {
          const randKey = keys[Math.floor(Math.random() * keys.length)];
          const item = next[randKey];
          const delta = (Math.random() - 0.48) * 0.02;
          const newPrice = +(Math.max(0.1, item.price_per_hour + delta)).toFixed(4);
          const newChange = +(item.change_percent + (delta > 0 ? 0.15 : -0.15)).toFixed(2);
          next[randKey] = {
            ...item,
            price_per_hour: newPrice,
            change_percent: newChange,
            bid: +(newPrice - 0.01).toFixed(4),
            ask: +(newPrice + 0.01).toFixed(4),
            volume_24h: item.volume_24h + Math.floor(Math.random() * 50),
          };
        }
        return next;
      });
    }, 2500);

    return () => clearInterval(interval);
  }, []);

  return { prices };
}
