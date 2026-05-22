// hooks/useResearchWebSocket.ts
import { useEffect, useRef, useState, useCallback } from 'react';
import { useAuthStore } from '@/stores/authStore';

interface WebSocketMessage {
  type: 'chunk' | 'end' | 'error';
  content?: string;
}

export function useResearchWebSocket() {
  const [isConnected, setIsConnected] = useState(false);
  const [response, setResponse] = useState('');
  const wsRef = useRef<WebSocket | null>(null);
  const { token } = useAuthStore();

  const connect = useCallback(() => {
    if (!token) return;
    const ws = new WebSocket(`ws://localhost:8000/ws/research/?token=${token}`);
    wsRef.current = ws;

    ws.onopen = () => setIsConnected(true);
    ws.onmessage = (event) => {
      const data: WebSocketMessage = JSON.parse(event.data);
      if (data.type === 'chunk') {
        setResponse(prev => prev + data.content);
      } else if (data.type === 'end') {
        ws.close();
      }
    };
    ws.onclose = () => setIsConnected(false);
    ws.onerror = () => setIsConnected(false);
  }, [token]);

  const sendMessage = useCallback((message: string) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      setResponse('');
      wsRef.current.send(JSON.stringify({ message }));
    } else {
      connect();
      setTimeout(() => sendMessage(message), 500);
    }
  }, [connect]);

  const disconnect = useCallback(() => {
    if (wsRef.current) wsRef.current.close();
  }, []);

  return { isConnected, response, sendMessage, disconnect };
}