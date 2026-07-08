import React from "react";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Badge } from "../../../components/ui/Badge";
import { MarkdownRenderer } from "../../community/components/MarkdownRenderer";

export function ComparisonTable({ result }: any) {
  return (
    <div className="grid md:grid-cols-2 gap-4">
      <TerminalCard className="p-4">
        <div className="flex justify-between">
          <h3 className="font-bold text-white">{result.model_a.model}</h3>
          <Badge variant="info">
            {result.tokens_a.input + result.tokens_a.output} tokens
          </Badge>
        </div>
        <div className="text-xs text-slate-400">
          Latency: {result.latency_ms_a}ms
        </div>
        <div className="mt-3 prose prose-invert max-w-none">
          <MarkdownRenderer content={result.response_a} />
        </div>
      </TerminalCard>
      <TerminalCard className="p-4">
        <div className="flex justify-between">
          <h3 className="font-bold text-white">{result.model_b.model}</h3>
          <Badge variant="info">
            {result.tokens_b.input + result.tokens_b.output} tokens
          </Badge>
        </div>
        <div className="text-xs text-slate-400">
          Latency: {result.latency_ms_b}ms
        </div>
        <div className="mt-3 prose prose-invert max-w-none">
          <MarkdownRenderer content={result.response_b} />
        </div>
      </TerminalCard>
    </div>
  );
}
