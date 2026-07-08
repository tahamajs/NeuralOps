import { useEffect, useState, useCallback, useRef } from "react";
import { MarketplaceListing, Message, ComplianceCheck } from "../types";

interface WebSocketMessage {
  type:
    | "token"
    | "listings"
    | "suggestion"
    | "compliance_update"
    | "error"
    | "done";
  data: string | MarketplaceListing[] | string[] | ComplianceCheck;
}

export function useMarketplaceAgentWebSocket(sessionId: string | null) {
  const [isConnected, setIsConnected] = useState(false);
  const [streamingContent, setStreamingContent] = useState("");
  const [liveListings, setLiveListings] = useState<MarketplaceListing[]>([]);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [complianceUpdates, setComplianceUpdates] = useState<
    Record<string, ComplianceCheck>
  >({});
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!sessionId) return;

    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(
      `${protocol}://${window.location.host}/ws/marketplace-ai/${sessionId}/`,
    );
    wsRef.current = ws;

    ws.onopen = () => setIsConnected(true);
    ws.onmessage = (event) => {
      const msg: WebSocketMessage = JSON.parse(event.data);
      switch (msg.type) {
        case "token":
          setStreamingContent((prev) => prev + (msg.data as string));
          break;
        case "listings":
          setLiveListings(msg.data as MarketplaceListing[]);
          break;
        case "suggestion":
          setSuggestions(msg.data as string[]);
          break;
        case "compliance_update":
          setComplianceUpdates((prev) => ({
            ...prev,
            [(msg.data as ComplianceCheck).listing_id]:
              msg.data as ComplianceCheck,
          }));
          break;
        case "error":
          setError(msg.data as string);
          break;
        case "done":
          setIsConnected(false);
          break;
      }
    };
    ws.onerror = () => setError("WebSocket connection error");
    ws.onclose = () => setIsConnected(false);

    return () => ws.close();
  }, [sessionId]);

  const sendMessage = useCallback((content: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "message", content }));
      setStreamingContent("");
      setSuggestions([]);
    }
  }, []);

  const requestCompliance = useCallback((listingId: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({ type: "compliance", listing_id: listingId }),
      );
    }
  }, []);

  const applyFilter = useCallback((filterType: string, value: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({ type: "filter", filter_type: filterType, value }),
      );
    }
  }, []);

  return {
    isConnected,
    streamingContent,
    liveListings,
    suggestions,
    complianceUpdates,
    error,
    sendMessage,
    requestCompliance,
    applyFilter,
  };
}
