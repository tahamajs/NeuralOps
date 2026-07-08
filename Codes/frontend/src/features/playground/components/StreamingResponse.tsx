import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Loader2 } from "lucide-react";
import ReactMarkdown from "react-markdown";

export function StreamingResponse({ content, isStreaming, error }: any) {
  return (
    <TerminalCard className="p-4 max-h-80 overflow-auto">
      <div className="text-sm text-slate-400 mb-1">Response:</div>
      <div className="prose prose-invert max-w-none">
        <ReactMarkdown>{content}</ReactMarkdown>
        {isStreaming && (
          <Loader2 size={14} className="animate-spin inline ml-2" />
        )}
        {error && <div className="text-red-400">Error: {error}</div>}
      </div>
    </TerminalCard>
  );
}
