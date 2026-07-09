// Full Agent Dashboard page — connects to /api/agents/ endpoints
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import {
  Bot,
  Plus,
  MessageSquare,
  FlaskConical,
  BookOpen,
  Wrench,
  Zap,
  Code2,
  Search,
  FileText,
  Clock,
  Activity,
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import AgentStatusBadge from "../components/AgentStatusBadge";
import type { ChatSession, AgentStatus } from "../types";

const QUICK_STARTS = [
  {
    icon: Search,
    color: "#818cf8",
    title: "Research Agent",
    desc: "Search papers, build citation graphs, synthesize literature",
    prompt: "Search for recent papers about efficient transformer attention mechanisms",
  },
  {
    icon: FlaskConical,
    color: "#34d399",
    title: "Experiment Agent",
    desc: "Configure, launch, and analyze ML training runs",
    prompt: "Configure a DistilBERT experiment on SST-2 with LoRA compression",
  },
  {
    icon: Code2,
    color: "#f59e0b",
    title: "Code Agent",
    desc: "Write, debug, and review Python ML code",
    prompt: "Help me implement a custom attention layer with Galerkin approximation",
  },
  {
    icon: FileText,
    color: "#f472b6",
    title: "Paper Agent",
    desc: "Draft, edit, and format LaTeX research papers",
    prompt: "Help me write the introduction section of my paper on SVD attention",
  },
];

export default function AgentDashboardPage() {
  const navigate = useNavigate();

  const { data: sessions, isLoading: sessionsLoading } = useQuery<ChatSession[]>({
    queryKey: ["agentSessions"],
    queryFn: async () => (await api.get("/agents/chats/")).data,
    refetchInterval: 10_000,
  });

  const { data: status } = useQuery<AgentStatus>({
    queryKey: ["agentStatus"],
    queryFn: async () => {
      try {
        return (await api.get("/agents/status/")).data;
      } catch {
        return { status: "offline", model: "Unknown", tools_count: 0, active_sessions: 0, ws_enabled: false } as AgentStatus;
      }
    },
    refetchInterval: 15_000,
  });

  const handleNewSession = async (initialPrompt?: string) => {
    const res = await api.post("/agents/chats/", { title: "" });
    const id: string = res.data.id;
    if (initialPrompt) {
      await api.post(`/api/agents/chats/${id}/send/`, { text: initialPrompt });
    }
    navigate(`/agent/${id}`);
  };

  const stats = [
    { label: "Total Sessions", value: sessions?.length ?? 0, icon: MessageSquare, color: "#818cf8" },
    { label: "Tools Available", value: status?.tools_count ?? 0, icon: Wrench, color: "#34d399" },
    { label: "Active Sessions", value: status?.active_sessions ?? 0, icon: Activity, color: "#f59e0b" },
    { label: "AI Model", value: status?.model ?? "—", icon: Zap, color: "#f472b6" },
  ];

  return (
    <div className="space-y-6 animate-fade-in max-w-5xl mx-auto">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-display font-bold text-white flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
              <Bot size={20} className="text-white" />
            </div>
            AI Research Agent
          </h1>
          <p className="text-slate-400 text-sm mt-1 ml-12">
            Your intelligent research and experiment assistant
          </p>
        </div>
        <div className="flex items-center gap-3">
          {status && (
            <AgentStatusBadge status={status.ws_enabled ? "open" : "closed"} />
          )}
          <Button onClick={() => handleNewSession()} size="sm">
            <Plus size={14} className="mr-1" /> New Chat
          </Button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {stats.map((stat) => (
          <Card key={stat.label} className="p-4 text-center">
            <div className="flex justify-center mb-2">
              <stat.icon size={20} style={{ color: stat.color }} />
            </div>
            <div className="text-xl font-bold text-white">{stat.value}</div>
            <div className="text-xs text-slate-400 mt-0.5">{stat.label}</div>
          </Card>
        ))}
      </div>

      {/* Quick starts */}
      <div>
        <h2 className="text-sm font-display font-bold text-slate-400 uppercase tracking-wider mb-3">
          Quick Start
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {QUICK_STARTS.map((qs) => (
            <button
              key={qs.title}
              onClick={() => handleNewSession(qs.prompt)}
              className="text-left glass border border-white/10 rounded-xl p-4 hover:border-white/20 hover:bg-white/5 transition-all duration-200 group"
            >
              <div className="flex items-start gap-3">
                <div
                  className="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0"
                  style={{ background: `${qs.color}22`, border: `1px solid ${qs.color}44` }}
                >
                  <qs.icon size={18} style={{ color: qs.color }} />
                </div>
                <div>
                  <div className="text-sm font-bold text-white group-hover:text-white/90">
                    {qs.title}
                  </div>
                  <div className="text-xs text-slate-400 mt-0.5">{qs.desc}</div>
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Recent sessions */}
      <div>
        <h2 className="text-sm font-display font-bold text-slate-400 uppercase tracking-wider mb-3">
          Recent Sessions
        </h2>
        {sessionsLoading ? (
          <div className="flex justify-center py-8"><Spinner /></div>
        ) : !sessions || sessions.length === 0 ? (
          <Card className="text-center py-12">
            <Bot size={32} className="text-slate-600 mx-auto mb-3" />
            <p className="text-slate-400">No sessions yet. Start a new chat above.</p>
          </Card>
        ) : (
          <div className="space-y-2">
            {sessions.slice(0, 10).map((session) => (
              <button
                key={session.id}
                onClick={() => navigate(`/agent/${session.id}`)}
                className="w-full text-left glass border border-white/10 rounded-xl px-4 py-3 hover:border-white/20 hover:bg-white/5 transition-all duration-200 flex items-center gap-3"
              >
                <div className="w-8 h-8 rounded-lg bg-indigo-500/20 border border-indigo-500/30 flex items-center justify-center flex-shrink-0">
                  <MessageSquare size={14} className="text-indigo-400" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-white truncate">
                    {session.title || "Untitled Session"}
                  </div>
                  <div className="text-xs text-slate-500 flex items-center gap-1 mt-0.5">
                    <Clock size={10} />
                    {formatDistanceToNow(new Date(session.updated_at), { addSuffix: true })}
                    {session.message_count > 0 && (
                      <span className="ml-2 text-slate-600">· {session.message_count} messages</span>
                    )}
                  </div>
                </div>
                <BookOpen size={14} className="text-slate-600 flex-shrink-0" />
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
