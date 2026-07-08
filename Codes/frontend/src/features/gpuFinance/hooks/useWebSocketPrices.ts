// src/features/gpuFinance/hooks/useWebSocketPrices.ts
import { useEffect, useState, useRef, useCallback } from "react";
import { useAuthStore } from "../../auth/store/authStore";

export function useWebSocketPrices(contractId: string | null) {
  const [prices, setPrices] = useState<{
    current: number;
    bid: number;
    ask: number;
  } | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const { token } = useAuthStore();

  const connect = useCallback(() => {
    if (!contractId) return;
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(
      `${protocol}://${window.location.host}/ws/gpu-finance/${contractId}/?token=${token}`,
    );
    wsRef.current = ws;
    ws.onopen = () => setIsConnected(true);
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setPrices(data);
    };
    ws.onclose = () => {
      setIsConnected(false);
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = setTimeout(connect, 3000); // ✅ proper reconnection
    };
    ws.onerror = () => setIsConnected(false);
  }, [contractId, token]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { prices, isConnected };
}
