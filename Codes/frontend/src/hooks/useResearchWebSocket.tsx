import { useState, useCallback } from "react";

export function useResearchWebSocket() {
  const [isConnected, setIsConnected] = useState(true);
  const [response, setResponse] = useState<string>("");

  const sendMessage = useCallback((msg: string) => {
    setResponse("");
    setTimeout(() => {
      setResponse(`AI response to research query: "${msg}"`);
    }, 1000);
  }, []);

  return { isConnected, response, sendMessage };
}

export default useResearchWebSocket;
