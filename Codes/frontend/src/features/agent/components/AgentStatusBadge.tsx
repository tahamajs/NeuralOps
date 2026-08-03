// AgentStatusBadge — shows WebSocket connection status with colored dot
import type { FC } from "react";

type WsStatus = "connecting" | "open" | "closed" | "error";

interface Props {
  status: WsStatus;
}

const CONFIG: Record<
  WsStatus,
  { color: string; label: string; pulse: boolean }
> = {
  open: { color: "#22c55e", label: "Live", pulse: true },
  connecting: { color: "#f59e0b", label: "Connecting", pulse: true },
  closed: { color: "#64748b", label: "Offline", pulse: false },
  error: { color: "#ef4444", label: "Error", pulse: false },
};

const AgentStatusBadge: FC<Props> = ({ status }) => {
  const cfg = CONFIG[status];
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        padding: "2px 10px",
        borderRadius: 999,
        background: "rgba(255,255,255,0.05)",
        border: "1px solid rgba(255,255,255,0.1)",
        fontSize: 11,
        fontWeight: 600,
        letterSpacing: "0.05em",
        color: cfg.color,
        backdropFilter: "blur(8px)",
      }}
    >
      <span
        style={{
          width: 7,
          height: 7,
          borderRadius: "50%",
          background: cfg.color,
          flexShrink: 0,
          boxShadow: cfg.pulse ? `0 0 6px ${cfg.color}` : "none",
          animation: cfg.pulse ? "pulse 1.5s ease-in-out infinite" : "none",
        }}
      />
      {cfg.label}
    </span>
  );
};

export default AgentStatusBadge;
