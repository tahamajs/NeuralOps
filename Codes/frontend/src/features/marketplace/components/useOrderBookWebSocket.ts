import { useEffect, useState, useRef, useCallback } from "react";
import { getAuthToken } from "../../../utils/authUtils";
import { OrderBookDepth } from "../types";

export function useOrderBookWebSocket(listingId: string | null) {
  const [liveDepth, setLiveDepth] = useState<OrderBookDepth | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const connect = useCallback(() => {
    if (!listingId) return;
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const token = getAuthToken();
    const ws = new WebSocket(
      `${protocol}://${window.location.host}/ws/marketplace/depth/${listingId}/?token=${token}`,
    );
    wsRef.current = ws;
    ws.onopen = () => setIsConnected(true);
    ws.onmessage = (event) => {
      try {
        setLiveDepth(JSON.parse(event.data));
      } catch (err) {
        console.error("OrderBook WS parse error", err);
      }
    };
    ws.onerror = () => setIsConnected(false);
    ws.onclose = () => {
      setIsConnected(false);
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = setTimeout(connect, 3000);
    };
  }, [listingId]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { liveDepth, isConnected };
}
