import React from "react";
import { Terminal, Copy, Check } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

interface StreamingOutputProps {
  content: string;
  isStreaming: boolean;
}

export function StreamingOutput({ content, isStreaming }: StreamingOutputProps) {
  const { success } = useToast();
  const [copied, setCopied] = React.useState(false);

  const handleCopy = () => {
    if (!content) return;
    navigator.clipboard.writeText(content);
    setCopied(true);
    success("Copied to clipboard");
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex flex-col h-full min-h-[160px] text-xs font-mono">
      <div className="flex justify-between items-center bg-slate-900 border-b border-white/10 px-3.5 py-2 rounded-t-xl text-slate-400">
        <span className="flex items-center gap-1.5 font-bold">
          <Terminal size={12} className="text-neon-blue animate-pulse" />
          Output Console
        </span>
        {content && (
          <button
            onClick={handleCopy}
            className="hover:text-white transition-colors p-1 hover:bg-white/5 rounded"
            title="Copy Output"
          >
            {copied ? <Check size={12} className="text-emerald-400" /> : <Copy size={12} />}
          </button>
        )}
      </div>

      <div className="flex-1 bg-slate-950 p-4 rounded-b-xl border border-t-0 border-white/10 overflow-y-auto whitespace-pre-wrap text-slate-200 leading-relaxed text-sm select-text">
        {content ? (
          <>
            {content}
            {isStreaming && (
              <span className="inline-block w-2 h-4 bg-neon-blue ml-0.5 animate-pulse" />
            )}
          </>
        ) : (
          <span className="text-slate-600 italic">Console output is empty. Run a prompt.</span>
        )}
      </div>
    </div>
  );
}
