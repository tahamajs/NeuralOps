// src/features/liveDashboards/hooks/useLiveDashboard.ts
import { useEffect, useState, useRef, useCallback, useMemo } from "react";
import { getAuthToken } from "../../../utils/authUtils";
import { Widget, LiveDataPoint } from "../types";

interface WebSocketMessage {
  type: "data" | "pong" | "error";
  widgetId: string;
  data: LiveDataPoint[];
}

import { liveDashboardsAPI } from "../services/api";

export function useLiveDashboard(
  dashboardId: string | null,
  widgets: Widget[],
  refreshInterval: number = 0,
) {
  const [liveData, setLiveData] = useState<Record<string, LiveDataPoint[]>>({});
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const heartbeatIntervalRef = useRef<ReturnType<typeof setInterval> | null>(
    null,
  );
  const widgetIdsKey = useMemo(() => widgets.map((w) => w.id).sort().join(","), [widgets]);

  const connect = useCallback(() => {
    if (!dashboardId) return;
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const token = getAuthToken();
    const ws = new WebSocket(
      `${protocol}://${window.location.host}/ws/live-dash/${dashboardId}/?token=${token}`,
    );
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      // Send subscription for all widgets
      const ids = widgetIdsKey ? widgetIdsKey.split(",") : [];
      ws.send(
        JSON.stringify({
          type: "subscribe",
          widgetIds: ids,
        }),
      );
      // Heartbeat every 30s
      if (heartbeatIntervalRef.current) clearInterval(heartbeatIntervalRef.current);
      heartbeatIntervalRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN)
          ws.send(JSON.stringify({ type: "ping" }));
      }, 30000);
    };

    ws.onmessage = (event) => {
      try {
        const msg: WebSocketMessage = JSON.parse(event.data);
        if (msg.type === "data") {
          setLiveData((prev) => ({ ...prev, [msg.widgetId]: msg.data }));
        } else if (msg.type === "error") {
          console.error("LiveDashboard WebSocket error:", msg);
        }
      } catch (err) {
        console.error("LiveDashboard WS parse error", err);
      }
    };

    ws.onerror = () => setIsConnected(false);

    ws.onclose = () => {
      setIsConnected(false);
      if (heartbeatIntervalRef.current)
        clearInterval(heartbeatIntervalRef.current);
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = setTimeout(connect, 3000);
    };
  }, [dashboardId, widgetIdsKey]);

  useEffect(() => {
    if (!dashboardId) return;
    connect();
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (heartbeatIntervalRef.current)
        clearInterval(heartbeatIntervalRef.current);
    };
  }, [connect, dashboardId]);

  // Fallback polling if refreshInterval > 0 and WebSocket not used (optional)
  useEffect(() => {
    if (refreshInterval <= 0 || !dashboardId) return;
    const interval = setInterval(() => {
      // Poll each widget individually (simplified)
      widgets.forEach((widget) => {
        liveDashboardsAPI
          .fetchWidgetData(widget.id, widget.data_source.params)
          .then((res) => setLiveData((prev) => ({ ...prev, [widget.id]: res })))
          .catch(console.error);
      });
    }, refreshInterval * 1000);
    return () => clearInterval(interval);
  }, [dashboardId, widgets, refreshInterval]);

  return { liveData, isConnected };
}
