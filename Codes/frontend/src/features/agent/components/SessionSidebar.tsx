import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { Button } from "../../../components/ui/Button";
import { Plus, MessageSquare, Trash2 } from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import { ChatSession } from "../types";

interface Props {
  activeId?: string;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
}

export default function SessionSidebar({
  activeId,
  onSelect,
  onNew,
  onDelete,
}: Props) {
  const { data: sessions = [] } = useQuery<ChatSession[]>({
    queryKey: ["agentSessions"],
    queryFn: async () => (await api.get("/api/agents/chats/")).data,
    refetchInterval: 10000,
  });

  return (
    <div className="w-64 glass border-r border-white/5 flex flex-col h-full">
      <div className="p-3 border-b border-white/5">
        <Button size="sm" className="w-full" onClick={onNew}>
          <Plus size={14} className="mr-1" /> New Chat
        </Button>
      </div>
      <div className="flex-1 overflow-y-auto scrollbar-thin px-2 py-2">
        {sessions.map((s) => (
          <div
            key={s.id}
            className={`group flex items-center px-3 py-2 rounded-lg cursor-pointer transition-colors mb-0.5 ${
              activeId === s.id
                ? "bg-neon-blue/10 text-white border border-neon-blue/20"
                : "text-slate-400 hover:bg-white/5 hover:text-white"
            }`}
            onClick={() => onSelect(s.id)}
          >
            <MessageSquare size={14} className="mr-2 flex-shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="text-xs font-medium truncate">
                {s.title || "New Chat"}
              </p>
              <p className="text-[10px] text-slate-500">
                {formatDistanceToNow(new Date(s.updated_at), {
                  addSuffix: true,
                })}
              </p>
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onDelete(s.id);
              }}
              className="p-1 opacity-0 group-hover:opacity-100 hover:bg-red-500/20 rounded transition-all"
            >
              <Trash2 size={12} className="text-slate-400 hover:text-red-400" />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
