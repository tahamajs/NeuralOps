import React from "react";
import { Info, Sparkles, BookOpen, AlertCircle } from "lucide-react";

interface ExplanationPanelProps {
  title?: string;
  explanation?: string;
  highlights?: string[];
  suggestions?: string[];
}

export function ExplanationPanel({
  title = "Optimization Insights",
  explanation = "Use the explanation tab to inspect model code complexity, attention heads configuration, or low-rank scaling factors recommendation.",
  highlights = [],
  suggestions = [],
}: ExplanationPanelProps) {
  return (
    <div className="border border-white/10 bg-slate-900/30 rounded-2xl p-5 backdrop-blur-md text-slate-200 font-sans text-xs space-y-4">
      <div className="flex items-center gap-2 border-b border-white/5 pb-3">
        <div className="p-1.5 bg-neon-blue/10 rounded-lg text-neon-blue border border-neon-blue/20">
          <BookOpen size={16} />
        </div>
        <h4 className="text-sm font-extrabold text-white tracking-tight">{title}</h4>
      </div>

      <div className="space-y-3">
        <p className="text-slate-400 leading-relaxed text-sm bg-white/5 p-4 rounded-xl border border-white/5 font-medium">
          {explanation}
        </p>

        {highlights.length > 0 && (
          <div className="space-y-2">
            <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block">Structural Details</span>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {highlights.map((h, i) => (
                <div key={i} className="flex items-start gap-2 bg-slate-950 p-2.5 rounded-lg border border-white/5">
                  <Sparkles size={12} className="text-purple-400 shrink-0 mt-0.5" />
                  <span className="text-slate-300 font-medium">{h}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {suggestions.length > 0 && (
          <div className="space-y-2">
            <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block">Recommendations</span>
            <div className="space-y-1.5">
              {suggestions.map((s, i) => (
                <div key={i} className="flex items-start gap-2 text-slate-400">
                  <AlertCircle size={12} className="text-neon-blue shrink-0 mt-0.5" />
                  <span className="leading-normal">{s}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
