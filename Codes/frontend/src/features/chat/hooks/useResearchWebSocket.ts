import { useEffect, useRef, useState, useCallback } from "react";
import { getAuthToken } from "../../../utils/authUtils";

type WebSocketStatus = "connecting" | "open" | "closing" | "closed" | "error";

interface WebSocketOptions {
  onMessage?: (data: any) => void;
  onStatusChange?: (status: WebSocketStatus) => void;
  onError?: (error: Event) => void;
  autoReconnect?: boolean;
  maxReconnectAttempts?: number;
  baseReconnectDelay?: number;
  maxReconnectDelay?: number;
}

export function useResearchWebSocket(
  sessionId: string | null,
  options: WebSocketOptions = {},
) {
  const {
    onMessage,
    onStatusChange,
    onError,
    autoReconnect = true,
    maxReconnectAttempts = 10,
    baseReconnectDelay = 1000,
    maxReconnectDelay = 30000,
  } = options;

  const token = getAuthToken();
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const messageQueueRef = useRef<any[]>([]);
  const [status, setStatus] = useState<WebSocketStatus>("closed");
  const [reconnectCountdown, setReconnectCountdown] = useState(0);

  const updateStatus = useCallback(
    (newStatus: WebSocketStatus) => {
      setStatus(newStatus);
      onStatusChange?.(newStatus);
    },
    [onStatusChange],
  );

  const flushMessageQueue = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      while (messageQueueRef.current.length) {
        const msg = messageQueueRef.current.shift();
        wsRef.current.send(JSON.stringify(msg));
      }
    }
  }, []);

  const sendMessage = useCallback((data: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    } else {
      // Buffer messages when disconnected
      messageQueueRef.current.push(data);
    }
  }, []);

  const connect = useCallback(() => {
    if (!sessionId) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    // Clear any pending reconnect timer
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }

    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const host =
      (import.meta as any).env?.VITE_WS_URL ||
      (window.location.port === "3000"
        ? `${window.location.hostname}:8000`
        : `${window.location.host}`);
    const wsUrl = `${protocol}://${host}/ws/chat/${sessionId}/?token=${token}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;
    updateStatus("connecting");

    ws.onopen = () => {
      updateStatus("open");
      reconnectAttemptsRef.current = 0;
      setReconnectCountdown(0);
      flushMessageQueue();

      // Send ping every 30 seconds to keep connection alive
      pingIntervalRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "ping" }));
        }
      }, 30000);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "pong") return;
        onMessage?.(data);
      } catch (err) {
        console.error("WebSocket message parse error", err);
      }
    };

    ws.onerror = (event) => {
      updateStatus("error");
      onError?.(event);
    };

    ws.onclose = () => {
      updateStatus("closed");
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
        pingIntervalRef.current = null;
      }

      if (
        autoReconnect &&
        reconnectAttemptsRef.current < maxReconnectAttempts
      ) {
        const delay = Math.min(
          baseReconnectDelay * Math.pow(2, reconnectAttemptsRef.current),
          maxReconnectDelay,
        );
        reconnectAttemptsRef.current++;
        setReconnectCountdown(Math.ceil(delay / 1000));

        reconnectTimerRef.current = setTimeout(() => {
          connect();
        }, delay);
      }
    };
  }, [
    sessionId,
    token,
    autoReconnect,
    maxReconnectAttempts,
    baseReconnectDelay,
    maxReconnectDelay,
    updateStatus,
    onMessage,
    onError,
    flushMessageQueue,
  ]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (pingIntervalRef.current) clearInterval(pingIntervalRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  return {
    status,
    reconnectCountdown,
    sendMessage,
    isConnected: status === "open",
  };
}
