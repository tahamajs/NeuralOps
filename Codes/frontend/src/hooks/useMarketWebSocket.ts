import { useState, useEffect } from "react";

export interface LiveMarketPrice {
  price_per_hour: number;
  change_percent: number;
}

export function useMarketWebSocket() {
  const [prices, setPrices] = useState<Record<string, LiveMarketPrice>>({});

  useEffect(() => {
    // Basic simulator for live prices on the marketplace
    const interval = setInterval(() => {
      setPrices((prev) => {
        const next = { ...prev };
        // Choose a random key or generate one if needed
        return next;
      });
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  return { prices };
}
