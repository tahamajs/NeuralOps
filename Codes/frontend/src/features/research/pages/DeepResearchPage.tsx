// src/features/research/pages/DeepResearchPage.tsx
import { useState } from "react";
import { useResearchStream } from "../hooks/useResearchStream";
import { ResearchTree } from "../components/ResearchTree";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Progress } from "../../../components/ui/Progress";
import { MarkdownRenderer } from "../../community/components/MarkdownRenderer";

export default function DeepResearchPage() {
  const [sessionId] = useState(() => crypto.randomUUID());
  const [query, setQuery] = useState("");
  const [isResearching, setIsResearching] = useState(false);
  const [tasks, setTasks] = useState<any[]>([]);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const { events, isActive } = useResearchStream(
    isResearching ? sessionId : "",
  );

  const startResearch = async () => {
    setIsResearching(true);
    await fetch("/api/research/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId, query }),
    });
  };

  const latestEvent = events[events.length - 1];
  const progress = latestEvent?.progress || { current: 0, total: 1 };
  const finalReport = events.find((e) => e.type === "final_report")?.report;

  return (
    <div className="flex h-full gap-4 p-4">
      {/* 左侧：任务树 */}
      <TerminalCard className="w-80 shrink-0 overflow-y-auto">
        <h3 className="terminal-heading mb-3">Research Plan</h3>
        <ResearchTree
          tasks={tasks}
          expanded={expanded}
          onToggle={setExpanded}
        />
      </TerminalCard>

      {/* 中侧：实时输出流 */}
      <TerminalCard className="flex-1 flex flex-col overflow-hidden">
        <div className="p-3 border-b border-white/10 flex justify-between items-center">
          <span className="text-sm font-mono text-neon-mint">
            ● {isActive ? "Researching" : "Idle"}
          </span>
          <Progress
            value={(progress.current / progress.total) * 100}
            className="w-32 h-1"
          />
        </div>
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          {events.map((ev, idx) => (
            <div key={idx} className="text-sm font-mono">
              {ev.type === "llm_token" ? (
                <span className="text-slate-300">{ev.token}</span>
              ) : (
                <div className="text-xs text-neon-blue">
                  [{ev.nodeId}] {ev.message}
                </div>
              )}
            </div>
          ))}
        </div>
      </TerminalCard>

      {/* 右侧：最终报告区 */}
      <TerminalCard className="w-96 shrink-0 overflow-y-auto">
        <h3 className="terminal-heading mb-3">Final Report</h3>
        {finalReport ? (
          <MarkdownRenderer content={finalReport} />
        ) : (
          <div className="text-slate-500 text-sm">研究报告将在此显示...</div>
        )}
      </TerminalCard>
    </div>
  );
}
