import { Link } from "react-router-dom";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { formatDistanceToNow } from "date-fns";

export function AlertDropdown({ alerts, onClose, onMarkAllRead }: any) {
  return (
    <div className="absolute right-0 mt-2 w-80 glass-terminal rounded-xl shadow-2xl z-50">
      <div className="flex justify-between items-center p-3 border-b border-white/10">
        <span className="text-sm font-bold text-white">Notifications</span>
        <button onClick={onMarkAllRead} className="text-xs text-neon-blue">
          Mark all read
        </button>
      </div>
      <div className="max-h-80 overflow-y-auto">
        {alerts.length === 0 ? (
          <div className="p-4 text-center text-slate-500">
            No new notifications
          </div>
        ) : (
          alerts.map((alert) => (
            <Link
              key={alert.id}
              to={alert.link}
              onClick={onClose}
              className={`block p-3 border-b border-white/5 hover:bg-white/5 ${!alert.read ? "bg-neon-blue/5" : ""}`}
            >
              <div className="text-sm text-white">{alert.message}</div>
              <div className="text-xs text-slate-400 mt-1">
                {formatDistanceToNow(new Date(alert.created_at), {
                  addSuffix: true,
                })}
              </div>
            </Link>
          ))
        )}
      </div>
    </div>
  );
}
