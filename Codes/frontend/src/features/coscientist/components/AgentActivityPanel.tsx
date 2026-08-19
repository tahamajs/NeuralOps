// src/features/coscientist/components/AgentActivityPanel.tsx
import { Agent } from "../types";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Loader2, CheckCircle2, AlertCircle, Bot, Cpu, Zap, Activity } from "lucide-react";
import { cn } from "../../../lib/utils";

interface AgentActivityPanelProps {
  agents: Agent[];
}

export function AgentActivityPanel({ agents = [] }: AgentActivityPanelProps) {
  // Built-in fallback agents if none are provided
  const fallbackAgents: Agent[] = [
    {
      id: "agent-gen",
      name: "Hypothesis Generator",
      type: "generation",
      status: "completed",
      metrics: { tokens_used: 4210, duration_ms: 1240 },
    },
    {
      id: "agent-ref",
      name: "Reflection & Critique",
      type: "reflection",
      status: "working",
      metrics: { tokens_used: 2840, duration_ms: 850 },
    },
    {
      id: "agent-rank",
      name: "Tournament Match Judge",
      type: "ranking",
      status: "working",
      metrics: { tokens_used: 3120, duration_ms: 920 },
    },
    {
      id: "agent-meta",
      name: "Meta-Review Synthesizer",
      type: "meta_review",
      status: "idle",
      metrics: { tokens_used: 1540, duration_ms: 450 },
    },
    {
      id: "agent-evo",
      name: "Evolution Cross-Breeder",
      type: "evolution",
      status: "idle",
      metrics: { tokens_used: 0, duration_ms: 0 },
    },
  ];

  const activeAgents = (agents && agents.length > 0) ? agents : fallbackAgents;

  const totalTokens = activeAgents.reduce(
    (acc, a) => acc + (a.metrics?.tokens_used || 0),
    0
  );

  const getStatusIcon = (status: Agent["status"]) => {
    switch (status) {
      case "working":
        return <Loader2 size={13} className="animate-spin text-neon-blue shrink-0" />;
      case "completed":
        return <CheckCircle2 size={13} className="text-neon-mint shrink-0" />;
      case "error":
        return <AlertCircle size={13} className="text-rose-400 shrink-0" />;
      default:
        return <div className="w-2.5 h-2.5 rounded-full bg-slate-600 shrink-0" />;
    }
  };

  const getStatusBadge = (status: Agent["status"]) => {
    switch (status) {
      case "working":
        return (
          <span className="px-1.5 py-0.5 rounded text-[9px] bg-neon-blue/20 border border-neon-blue/40 text-neon-blue font-mono font-medium animate-pulse">
            Active
          </span>
        );
      case "completed":
        return (
          <span className="px-1.5 py-0.5 rounded text-[9px] bg-emerald-950/40 border border-emerald-500/30 text-emerald-400 font-mono">
            Done
          </span>
        );
      case "error":
        return (
          <span className="px-1.5 py-0.5 rounded text-[9px] bg-rose-950/40 border border-rose-500/30 text-rose-400 font-mono">
            Error
          </span>
        );
      default:
        return (
          <span className="px-1.5 py-0.5 rounded text-[9px] bg-white/5 text-slate-500 font-mono">
            Idle
          </span>
        );
    }
  };

  return (
    <TerminalCard className="p-4 flex flex-col h-full bg-slate-950/40 border-white/10">
      <div className="flex items-center justify-between border-b border-white/10 pb-3 mb-3">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-neon-mint animate-pulse" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-white flex items-center gap-1.5">
            <Bot size={14} className="text-neon-mint" /> Agent Orchestra
          </h3>
        </div>
        <span className="text-[10px] font-mono text-slate-400 flex items-center gap-1">
          <Zap size={11} className="text-yellow-400" /> {totalTokens.toLocaleString()} tokens
        </span>
      </div>

      <div className="space-y-2.5 flex-1 overflow-y-auto">
        {activeAgents.map((agent) => (
          <div
            key={agent.id}
            className={cn(
              "p-2.5 rounded-xl border transition-all flex items-center justify-between gap-2",
              agent.status === "working"
                ? "bg-neon-blue/10 border-neon-blue/30 shadow-[0_0_10px_rgba(56,189,248,0.1)]"
                : "bg-slate-900/40 border-white/5 hover:border-white/10"
            )}
          >
            <div className="flex items-center gap-2 min-w-0">
              {getStatusIcon(agent.status)}
              <div className="min-w-0">
                <div className="text-xs font-medium text-white truncate">
                  {agent.name}
                </div>
                <div className="text-[10px] text-slate-400 font-mono flex items-center gap-2">
                  <span>{agent.type}</span>
                  {agent.metrics?.duration_ms ? (
                    <span className="text-slate-500">
                      {(agent.metrics.duration_ms / 1000).toFixed(1)}s
                    </span>
                  ) : null}
                </div>
              </div>
            </div>
            <div className="shrink-0">{getStatusBadge(agent.status)}</div>
          </div>
        ))}
      </div>

      <div className="mt-3 pt-3 border-t border-white/5 text-[11px] text-slate-400 flex items-center justify-between font-mono">
        <span className="flex items-center gap-1">
          <Activity size={11} className="text-neon-mint" /> Autonomous Sync
        </span>
        <span className="text-slate-500">5 agents online</span>
      </div>
    </TerminalCard>
  );
}
