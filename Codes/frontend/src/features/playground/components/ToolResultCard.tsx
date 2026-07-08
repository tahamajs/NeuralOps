import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Badge } from "../../../components/ui/Badge";

export function ToolResultCard({ result }: any) {
  return (
    <TerminalCard className="p-4">
      <div className="flex justify-between">
        <h3 className="font-bold text-white">{result.tool_name}</h3>
        <Badge variant={result.success ? "success" : "failed"}>
          {result.success ? "Success" : "Failed"}
        </Badge>
      </div>
      <div className="text-xs text-slate-400 mt-1">
        Execution time: {result.execution_time_ms}ms
      </div>
      <div className="mt-2">
        <div className="text-xs text-slate-400">Input:</div>
        <pre className="bg-black/30 p-2 rounded text-sm">
          {JSON.stringify(result.input, null, 2)}
        </pre>
      </div>
      <div className="mt-2">
        <div className="text-xs text-slate-400">Output:</div>
        <pre className="bg-black/30 p-2 rounded text-sm">
          {typeof result.output === "string"
            ? result.output
            : JSON.stringify(result.output, null, 2)}
        </pre>
      </div>
      {result.error && (
        <div className="mt-2 text-red-400 text-sm">{result.error}</div>
      )}
    </TerminalCard>
  );
}
