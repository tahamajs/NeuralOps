import { useEffect, useState, useRef, useCallback } from "react";
import { getAuthToken } from "../../../utils/authUtils";

export function useSocialWebSocket() {
  const [newAlert, setNewAlert] = useState<any>(null);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const connect = useCallback(() => {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const token = getAuthToken();
    const ws = new WebSocket(
      `${protocol}://${window.location.host}/ws/social/alerts/?token=${token}`,
    );
    wsRef.current = ws;
    ws.onopen = () => setIsConnected(true);
    ws.onmessage = (event) => {
      try {
        setNewAlert(JSON.parse(event.data));
      } catch (err) {
        console.error("Social WS parse error", err);
      }
    };
    ws.onerror = () => setIsConnected(false);
    ws.onclose = () => {
      setIsConnected(false);
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = setTimeout(connect, 3000);
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { newAlert, isConnected };
}
