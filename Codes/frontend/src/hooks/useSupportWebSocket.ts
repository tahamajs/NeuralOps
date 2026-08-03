import { useState, useCallback } from "react";

export interface SupportProblem {
  title: string;
  description: string;
  code?: string;
  category?: string;
  severity?: string;
}

export function useSupportWebSocket() {
  const [isConnected, setIsConnected] = useState(true);
  const [response, setResponse] = useState<string>("");
  const [isThinking, setIsThinking] = useState(false);
  const [steps, setSteps] = useState<string[]>([]);

  const sendMessage = useCallback((msg: string) => {
    setIsThinking(true);
    setResponse("");
    setSteps(["Parsing user query", "Searching internal knowledge base"]);

    // Simulate AI response stream
    setTimeout(() => {
      setSteps((prev) => [...prev, "Synthesizing explanation"]);
      setTimeout(() => {
        setIsThinking(false);
        setResponse(
          `Here is how you can approach your query: **${msg}**\n\n1. Check your network configuration.\n2. Ensure your backend services are running.\n3. Verify your API keys.`,
        );
      }, 1000);
    }, 1000);
  }, []);

  const submitProblem = useCallback((problem: SupportProblem) => {
    setIsThinking(true);
    setSteps(["Submitting support ticket", "Analyzing code snippet"]);
    setTimeout(() => {
      setIsThinking(false);
      setResponse(
        `### Support Ticket Received\n\n**Title**: ${problem.title}\n**Severity**: ${problem.severity}\n\nOur team has received this report and is analyzing the code. You will receive updates here.`,
      );
    }, 1500);
  }, []);

  return {
    isConnected,
    response,
    isThinking,
    steps,
    sendMessage,
    submitProblem,
  };
}
