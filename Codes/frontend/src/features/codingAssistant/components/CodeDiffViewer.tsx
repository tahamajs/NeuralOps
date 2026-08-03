import React from "react";
import { Check, Copy } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

interface CodeDiffViewerProps {
  diffText?: string;
  originalCode?: string;
  modifiedCode?: string;
}

export function CodeDiffViewer({
  diffText = "",
  originalCode = "",
  modifiedCode = "",
}: CodeDiffViewerProps) {
  const { success } = useToast();
  const [copied, setCopied] = React.useState(false);

  const handleCopy = () => {
    const textToCopy = modifiedCode || diffText;
    if (!textToCopy) return;
    navigator.clipboard.writeText(textToCopy);
    setCopied(true);
    success("Copied code to clipboard");
    setTimeout(() => setCopied(false), 2000);
  };

  // Helper to parse simple git diff lines
  const renderDiffLines = () => {
    if (diffText) {
      return diffText.split("\n").map((line, idx) => {
        let lineClass = "text-slate-300";
        if (line.startsWith("+")) {
          lineClass =
            "text-emerald-400 bg-emerald-950/20 px-1 border-l-2 border-emerald-500";
        } else if (line.startsWith("-")) {
          lineClass =
            "text-rose-400 bg-rose-950/20 px-1 border-l-2 border-rose-500 line-through";
        }
        return (
          <div
            key={idx}
            className={`font-mono text-xs py-0.5 whitespace-pre-wrap ${lineClass}`}
          >
            {line}
          </div>
        );
      });
    }

    // Side-by-side comparison fallback or simple split comparison
    if (originalCode && modifiedCode) {
      return (
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-0.5">
            <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block mb-2">
              Original
            </span>
            <pre className="bg-slate-950 p-3 rounded-xl border border-white/5 font-mono text-xs text-rose-300 overflow-x-auto whitespace-pre max-h-96">
              {originalCode}
            </pre>
          </div>
          <div className="space-y-0.5">
            <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block mb-2">
              Modified
            </span>
            <pre className="bg-slate-950 p-3 rounded-xl border border-white/5 font-mono text-xs text-emerald-300 overflow-x-auto whitespace-pre max-h-96">
              {modifiedCode}
            </pre>
          </div>
        </div>
      );
    }

    return (
      <div className="text-slate-500 italic">No code differences loaded.</div>
    );
  };

  return (
    <div className="border border-white/10 bg-slate-900/30 rounded-2xl overflow-hidden backdrop-blur-md text-slate-200">
      <div className="flex justify-between items-center bg-slate-950/40 border-b border-white/10 px-4 py-3">
        <span className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-neon-blue animate-pulse" />
          Code Diff Analyzer
        </span>
        {(modifiedCode || diffText) && (
          <button
            onClick={handleCopy}
            className="hover:text-white transition-colors p-1 hover:bg-white/5 rounded text-xs text-slate-400 flex items-center gap-1"
          >
            {copied ? (
              <Check size={12} className="text-emerald-400" />
            ) : (
              <Copy size={12} />
            )}
            Copy Code
          </button>
        )}
      </div>
      <div className="p-4 bg-slate-950/20 max-h-[400px] overflow-y-auto space-y-1">
        {renderDiffLines()}
      </div>
    </div>
  );
}
