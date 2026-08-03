import React, { useState } from "react";
import { EvaluationResult } from "../types";
import { Badge } from "../../../components/ui/Badge";
import { Cpu, DollarSign, Clock, ChevronDown, ChevronUp } from "lucide-react";

interface EvaluationResultCardProps {
  result: EvaluationResult;
}

export function EvaluationResultCard({ result }: EvaluationResultCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="border border-white/10 bg-slate-950/40 rounded-xl p-4 space-y-3 font-sans text-xs">
      <div className="flex justify-between items-center">
        <h4 className="text-sm font-extrabold text-white flex items-center gap-1.5">
          <Cpu size={14} className="text-neon-blue" />
          {result.model}
        </h4>
        <div className="flex gap-2">
          <span className="flex items-center gap-0.5 text-slate-400">
            <Clock size={12} /> {result.latency_ms}ms
          </span>
          <span className="flex items-center gap-0.5 text-slate-400">
            <DollarSign size={12} /> {result.cost_usd.toFixed(5)}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-white/5 p-3 rounded-lg border border-white/5">
        {Object.entries(result.scores || {}).map(([metric, score]) => (
          <div key={metric}>
            <span className="text-slate-500 block uppercase tracking-wider text-[9px] mb-0.5">
              {metric}
            </span>
            <div className="flex items-center gap-2">
              <div className="flex-1 bg-slate-900 h-1.5 rounded-full overflow-hidden">
                <div
                  className="bg-neon-blue h-full"
                  style={{ width: `${(score as number) * 100}%` }}
                />
              </div>
              <span className="font-mono font-bold text-white text-[10px]">
                {((score as number) * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        ))}
      </div>

      <div className="flex justify-between items-center text-[10px] text-slate-500">
        <span>
          Tokens:{" "}
          <strong className="text-slate-300 font-mono">
            {result.token_usage.input}
          </strong>{" "}
          in /{" "}
          <strong className="text-slate-300 font-mono">
            {result.token_usage.output}
          </strong>{" "}
          out
        </span>
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-neon-blue hover:text-white transition-colors flex items-center gap-0.5"
        >
          {expanded ? "Hide Output" : "Show Output"}{" "}
          {expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        </button>
      </div>

      {expanded && (
        <div className="bg-slate-950 p-3 rounded-lg border border-white/5 text-slate-300 font-mono text-[11px] whitespace-pre-wrap leading-normal overflow-x-auto max-h-48 select-text">
          {result.output || "No output captured."}
        </div>
      )}
    </div>
  );
}
