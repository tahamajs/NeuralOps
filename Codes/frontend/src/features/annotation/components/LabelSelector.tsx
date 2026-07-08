import { Badge } from "../../../components/ui/Badge";
import { Label } from "../types";

export function LabelSelector({ labels, selected, onSelect }: any) {
  return (
    <div className="flex flex-wrap gap-2">
      {labels.map((l: Label) => (
        <button
          key={l.id}
          onClick={() => onSelect(l.id)}
          className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${selected === l.id ? "bg-neon-blue text-white" : "bg-white/10 text-slate-300 hover:bg-white/20"}`}
          style={{ borderLeft: `3px solid ${l.color}` }}
        >
          {l.name}
          {l.shortcut && (
            <span className="ml-1 text-xs text-slate-400">({l.shortcut})</span>
          )}
        </button>
      ))}
    </div>
  );
}
