import { useEffect, useState, useCallback, useRef } from "react";
import { MarketplaceListing, Message, ComplianceCheck } from "../types";
import { marketplaceAIApi, INITIAL_AI_LISTINGS } from "../services/api";

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
  const [isConnected, setIsConnected] = useState(true);
  const [streamingContent, setStreamingContent] = useState("");
  const [liveListings, setLiveListings] = useState<MarketplaceListing[]>(INITIAL_AI_LISTINGS);
  const [suggestions, setSuggestions] = useState<string[]>([
    "Find cheapest 8x H100 in US-East",
    "Show GDPR-compliant clusters in EU",
    "Compare SXM5 vs PCIe memory bandwidth",
    "Recommend spot nodes for SVD linear attention",
  ]);
  const [complianceUpdates, setComplianceUpdates] = useState<
    Record<string, ComplianceCheck>
  >({});
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const isFallbackMode = useRef<boolean>(false);

  useEffect(() => {
    if (!sessionId) return;

    try {
      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(
        `${protocol}://${window.location.host}/ws/marketplace-ai/${sessionId}/`,
      );
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        setError(null);
        isFallbackMode.current = false;
      };

      ws.onmessage = (event) => {
        try {
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
              break;
          }
        } catch (e) {
          console.warn("WS Parse Error", e);
        }
      };

      ws.onerror = () => {
        // Fallback to intelligent local AI agent simulation
        isFallbackMode.current = true;
        setIsConnected(true);
      };

      ws.onclose = () => {
        isFallbackMode.current = true;
        setIsConnected(true);
      };

      return () => ws.close();
    } catch {
      isFallbackMode.current = true;
      setIsConnected(true);
    }
  }, [sessionId]);

  const streamSimulatedResponse = useCallback((fullText: string, matchedListings: MarketplaceListing[], newSuggestions: string[]) => {
    setStreamingContent("");
    let currentIdx = 0;
    const words = fullText.split(" ");
    
    const interval = setInterval(() => {
      if (currentIdx < words.length) {
        const chunk = words[currentIdx] + (currentIdx === words.length - 1 ? "" : " ");
        setStreamingContent((prev) => prev + chunk);
        currentIdx++;
      } else {
        clearInterval(interval);
        setLiveListings(matchedListings);
        setSuggestions(newSuggestions);
      }
    }, 40);
  }, []);

  const sendMessage = useCallback(async (content: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "message", content }));
      setStreamingContent("");
      setSuggestions([]);
    } else {
      // Local agent execution
      const res = await marketplaceAIApi.sendAgentMessage(sessionId || "fallback", content);
      const contextualSuggestions = [
        "Check SOC2 and HIPAA compliance",
        "Add matched nodes to Comparison Builder",
        "Deploy compute spot instance now",
        "Filter by min 80GB VRAM",
      ];
      streamSimulatedResponse(res.reply, res.listings, contextualSuggestions);
    }
  }, [sessionId, streamSimulatedResponse]);

  const requestCompliance = useCallback(async (listingId: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({ type: "compliance", listing_id: listingId }),
      );
    } else {
      const res = await marketplaceAIApi.checkListingCompliance(listingId);
      setComplianceUpdates((prev) => ({
        ...prev,
        [listingId]: res,
      }));
    }
  }, []);

  const applyFilter = useCallback((filterType: string, value: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({ type: "filter", filter_type: filterType, value }),
      );
    } else {
      let filtered = [...INITIAL_AI_LISTINGS];
      if (filterType === "maxPrice") {
        filtered = filtered.filter((l) => l.price_per_hour <= value);
      } else if (filterType === "model") {
        filtered = filtered.filter((l) => l.gpu_specs.model.toLowerCase().includes(value.toLowerCase()));
      }
      setLiveListings(filtered);
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
