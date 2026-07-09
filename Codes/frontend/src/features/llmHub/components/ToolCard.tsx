import React, { useState } from "react";
import { ToolDefinition } from "../types";
import { Switch } from "../../../components/ui/Switch";
import { Code, ChevronDown, ChevronUp } from "lucide-react";

interface ToolCardProps {
  tool: ToolDefinition;
  onToggle: (enabled: boolean) => void;
}

export function ToolCard({ tool, onToggle }: ToolCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="border border-white/10 bg-slate-950/40 rounded-xl p-4 space-y-3 font-sans text-xs">
      <div className="flex justify-between items-start gap-4">
        <div className="flex-1 min-w-0">
          <h4 className="text-sm font-extrabold text-white truncate">{tool.name}</h4>
          <p className="text-slate-400 text-xs mt-1 leading-normal">
            {tool.description}
          </p>
        </div>
        <div className="shrink-0">
          <Switch
            checked={tool.enabled}
            onChange={onToggle}
          />
        </div>
      </div>

      <div className="flex justify-between items-center pt-2 border-t border-white/5">
        <span className="text-[10px] text-slate-500 font-mono">
          Params: {Object.keys(tool.parameters?.properties || {}).length || 0} fields
        </span>
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-neon-blue hover:text-white transition-colors flex items-center gap-0.5 text-[10px]"
        >
          {expanded ? "Hide Schema" : "Show Schema"}{" "}
          {expanded ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
        </button>
      </div>

      {expanded && (
        <div className="space-y-2 mt-2 pt-2 border-t border-white/5">
          <div>
            <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block mb-1">Parameters Schema</span>
            <pre className="bg-slate-950 p-2.5 rounded-lg border border-white/5 font-mono text-[10px] text-slate-300 overflow-x-auto max-h-36 whitespace-pre">
              {JSON.stringify(tool.parameters || {}, null, 2)}
            </pre>
          </div>
          {tool.example && (
            <div>
              <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block mb-1">Example Payload</span>
              <pre className="bg-slate-950 p-2.5 rounded-lg border border-white/5 font-mono text-[10px] text-neon-mint overflow-x-auto max-h-24 whitespace-pre">
                {tool.example}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
