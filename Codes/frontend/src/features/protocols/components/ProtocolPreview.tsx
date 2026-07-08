import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Badge } from "../../../components/ui/Badge";
import ReactMarkdown from "react-markdown";

export function ProtocolPreview({ protocol }: any) {
  return (
    <div className="space-y-4">
      <TerminalCard className="p-4">
        <h2 className="text-xl font-bold text-white">{protocol.name}</h2>
        <div className="text-slate-300 mt-2 whitespace-pre-wrap">
          {protocol.description}
        </div>
        <div className="flex flex-wrap gap-1 mt-2">
          {protocol.tags?.map((t: string) => (
            <Badge key={t}>{t}</Badge>
          ))}
        </div>
      </TerminalCard>
      {protocol.steps?.map((step: any, idx: number) => (
        <TerminalCard key={step.id} className="p-4">
          <h3 className="text-lg font-bold text-white mb-2">
            Step {idx + 1}: {step.title}
          </h3>
          <div className="prose prose-invert max-w-none">
            <ReactMarkdown>{step.description}</ReactMarkdown>
          </div>
          <div className="text-xs text-slate-400 mt-2">Type: {step.type}</div>
        </TerminalCard>
      ))}
      {protocol.resources?.length > 0 && (
        <TerminalCard className="p-4">
          <h3 className="text-sm font-bold text-white mb-2">
            Required Resources
          </h3>
          <ul className="list-disc list-inside space-y-1 text-sm text-slate-300">
            {protocol.resources.map((r: any) => (
              <li key={r.id}>
                {r.name} ({r.type})
                {r.url && (
                  <a
                    href={r.url}
                    target="_blank"
                    className="text-neon-blue ml-2"
                  >
                    →
                  </a>
                )}
              </li>
            ))}
          </ul>
        </TerminalCard>
      )}
    </div>
  );
}
