import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { socialAPI } from "../services/api";
import { useSocialWebSocket } from "../hooks/useSocialWebSocket";
import { AlertDropdown } from "./AlertDropdown";
import { Bell } from "lucide-react";

export function AlertBell() {
  const [open, setOpen] = useState(false);
  const queryClient = useQueryClient();
  const { data: alerts, isLoading } = useQuery({
    queryKey: ["alerts"],
    queryFn: () => socialAPI.getAlerts(),
    refetchInterval: 30000,
  });
  const { newAlert, isConnected } = useSocialWebSocket();
  const unreadCount = alerts?.filter((a) => !a.read).length || 0;

  const markAllRead = useMutation({
    mutationFn: () => socialAPI.markAllAlertsRead(),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["alerts"] }),
  });

  if (isLoading) return null;

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="relative p-2 rounded-lg hover:bg-white/5 transition-colors"
      >
        <Bell
          size={18}
          className={isConnected ? "text-neon-mint" : "text-slate-400"}
        />
        {unreadCount > 0 && (
          <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full" />
        )}
      </button>
      {open && (
        <AlertDropdown
          alerts={alerts || []}
          onClose={() => setOpen(false)}
          onMarkAllRead={() => markAllRead.mutate()}
        />
      )}
    </div>
  );
}
