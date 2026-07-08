import { useState, useEffect } from "react";
export function useResearchStream(sessionId: string) {
  const [events, setEvents] = useState<any[]>([]);
  const [isActive, setIsActive] = useState(false);
  return { events, isActive };
}
