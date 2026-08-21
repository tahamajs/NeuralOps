// src/features/research/pages/DeepResearchPage.tsx
import { useState } from "react";
import { useResearchStream } from "../hooks/useResearchStream";
import { ResearchTree } from "../components/ResearchTree";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Progress } from "../../../components/ui/Progress";
import { MarkdownRenderer } from "../../community/components/MarkdownRenderer";
import {
  Brain,
  Search,
  FileText,
  Copy,
  Check,
  Download,
  Sparkles,
  Zap,
  Terminal,
  Activity,
  Layers,
  StopCircle,
} from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function DeepResearchPage({ onClose }: { onClose?: () => void }) {
  const [sessionId, setSessionId] = useState(() => crypto.randomUUID());
  const [query, setQuery] = useState(
    "Singular Value Decomposition (SVD) for Low-Rank Linear Attention in Long-Context Transformers"
  );
  const [depth, setDepth] = useState("deep");
  const [isResearching, setIsResearching] = useState(false);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({ "1": true, "2": true, "3": true });
  const [copied, setCopied] = useState(false);
  const { success, error } = useToast();

  const { events, isActive } = useResearchStream(isResearching ? sessionId : "");

  const startResearch = async () => {
    if (!query.trim()) return;
    const newSessionId = crypto.randomUUID();
    setSessionId(newSessionId);
    setIsResearching(true);
    success("Deep Research Started", "Agent swarm is scouring literature and synthesizing results.");
    try {
      await fetch("/api/research/start/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: newSessionId, query, depth }),
      });
    } catch (e) {
      // Stream hook handles local simulation seamlessly
    }
  };

  const stopResearch = () => {
    setIsResearching(false);
    success("Research Stopped");
  };

  const latestEvent = events[events.length - 1];
  const progress = latestEvent?.progress || {
    current: isResearching ? (isActive ? 6 : 10) : 0,
    total: 10,
  };
  const progressPercent = Math.min(100, Math.round((progress.current / progress.total) * 100));

  const finalReport = events.find((e) => e.type === "final_report")?.report;

  
  const displayReport = finalReport || "";

  const handleCopyReport = () => {
    if (!displayReport) return;
    navigator.clipboard.writeText(displayReport);
    setCopied(true);
    success("Report copied to clipboard");
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadReport = () => {
    if (!displayReport) return;
    const blob = new Blob([displayReport], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "Deep_Research_Synthesis_Report.md";
    a.click();
    URL.revokeObjectURL(url);
    success("Report downloaded");
  };

  return (
    <div className="flex flex-col h-full bg-slate-950/20">
      {/* Header & Controls */}
      <div className="p-4 border-b border-white/10 space-y-3">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-2.5">
            <div className="p-1.5 rounded-lg bg-neon-blue/10 text-neon-blue border border-neon-blue/20">
              <Brain size={20} />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white flex items-center gap-2">
                Deep Research Agent
              </h1>
              <p className="text-xs text-slate-400">
                Autonomous literature discovery, mathematical verification, and manuscript synthesis
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {onClose && (
              <Button variant="ghost" size="sm" onClick={onClose}>
                ✕
              </Button>
            )}
          </div>
        </div>

        {/* Search & Topic Bar */}
        <div className="flex flex-wrap gap-2.5">
          <div className="relative flex-1 min-w-[280px]">
            <Search
              size={15}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"
            />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter research topic, question, or hypothesis to synthesize..."
              className="pl-9 bg-slate-900/60 text-sm"
              onKeyDown={(e) => e.key === "Enter" && startResearch()}
            />
          </div>

          <Select
            options={[
              { value: "quick", label: "Quick Scan (1-2m)" },
              { value: "deep", label: "Deep Synthesis (Full)" },
              { value: "meta", label: "Exhaustive Meta-Review" },
            ]}
            value={depth}
            onChange={(e) => setDepth(e.target.value)}
            className="w-48"
          />

          {isResearching && isActive ? (
            <Button
              onClick={stopResearch}
              variant="outline"
              className="border-rose-500/40 text-rose-400 hover:bg-rose-500/10 gap-1.5"
            >
              <StopCircle size={15} /> Stop
            </Button>
          ) : (
            <Button
              onClick={startResearch}
              disabled={!query.trim()}
              className="bg-neon-blue hover:bg-neon-blue/80 text-white font-bold gap-1.5 px-5"
            >
              <Sparkles size={15} /> Start Deep Research
            </Button>
          )}
        </div>
      </div>

      {/* Main Content Area: 3 Columns */}
      <div className="flex-1 overflow-hidden p-4">
        <div className="grid lg:grid-cols-12 gap-4 h-full">
          {/* Left Column: Research Plan Tree (3 cols) */}
          <TerminalCard className="lg:col-span-3 h-full overflow-hidden flex flex-col p-3 bg-slate-950/40">
            <div className="flex items-center justify-between border-b border-white/10 pb-2.5 mb-2">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                <Layers size={13} className="text-neon-mint" /> Research Plan
              </h3>
              <span className="text-[10px] font-mono text-slate-400">
                {isResearching ? (isActive ? "IN PROGRESS" : "COMPLETE") : "READY"}
              </span>
            </div>
            <div className="flex-1 overflow-y-auto pr-1">
              <ResearchTree
                tasks={[]}
                expanded={expanded}
                onToggle={setExpanded}
              />
            </div>
          </TerminalCard>

          {/* Middle Column: Real-time Output Stream (4 cols) */}
          <TerminalCard className="lg:col-span-4 h-full flex flex-col overflow-hidden bg-slate-950/40">
            <div className="p-3 border-b border-white/10 flex justify-between items-center bg-slate-900/30">
              <span className="text-xs font-mono flex items-center gap-2">
                <span
                  className={`w-2 h-2 rounded-full ${
                    isActive ? "bg-neon-mint animate-pulse" : "bg-slate-500"
                  }`}
                />
                <span className={isActive ? "text-neon-mint font-bold" : "text-slate-400"}>
                  {isActive ? "Swarm Executing..." : "Agent Stream"}
                </span>
              </span>
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-mono text-slate-400">
                  {progressPercent}%
                </span>
                <Progress value={progressPercent} className="w-20 h-1.5" />
              </div>
            </div>

            <div className="flex-1 overflow-y-auto p-3.5 space-y-2 font-mono text-xs">
              {events.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-slate-500 text-center space-y-2">
                  <Terminal size={24} className="text-slate-600" />
                  <p className="text-xs max-w-xs">
                    Click <strong>Start Deep Research</strong> to launch autonomous literature searches and mathematical proofs.
                  </p>
                </div>
              ) : (
                events.map((ev, idx) => (
                  <div key={idx} className="leading-relaxed">
                    {ev.type === "llm_token" ? (
                      <span className="text-slate-200">{ev.token}</span>
                    ) : ev.type === "final_report" ? (
                      <div className="p-2 rounded bg-emerald-950/30 border border-emerald-500/30 text-emerald-400 text-xs">
                        ✓ Final Research Synthesis Ready
                      </div>
                    ) : (
                      <div className="p-1.5 rounded bg-white/5 border-l-2 border-neon-blue text-[11px] text-slate-300 my-1">
                        <span className="text-neon-blue font-bold font-mono">
                          [{ev.nodeId || "agent"}]
                        </span>{" "}
                        {ev.message}
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          </TerminalCard>

          {/* Right Column: Final Synthesized Report (5 cols) */}
          <TerminalCard className="lg:col-span-5 h-full overflow-hidden flex flex-col bg-slate-950/40">
            <div className="flex justify-between items-center p-3 border-b border-white/10 bg-slate-900/30">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                <FileText size={13} className="text-yellow-400" /> Synthesized Manuscript
              </h3>
              {displayReport && (
                <div className="flex items-center gap-1">
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={handleCopyReport}
                    className="text-xs text-slate-400 hover:text-white h-7 px-2"
                  >
                    {copied ? <Check size={12} className="text-neon-mint" /> : <Copy size={12} />}
                    <span className="ml-1 text-[10px]">{copied ? "Copied" : "Copy"}</span>
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={handleDownloadReport}
                    className="text-xs text-slate-400 hover:text-white h-7 px-2"
                  >
                    <Download size={12} />
                    <span className="ml-1 text-[10px]">Export</span>
                  </Button>
                </div>
              )}
            </div>

            <div className="flex-1 overflow-y-auto p-4 prose prose-invert max-w-none text-xs leading-relaxed">
              {displayReport ? (
                <MarkdownRenderer content={displayReport} />
              ) : (
                <div className="h-full flex flex-col items-center justify-center text-slate-500 text-center space-y-2">
                  <Brain size={28} className="text-slate-600" />
                  <p className="text-xs max-w-xs">
                    Synthesized findings, LaTeX equations, and benchmarks will appear here upon completion.
                  </p>
                </div>
              )}
            </div>
          </TerminalCard>
        </div>
      </div>
    </div>
  );
}
