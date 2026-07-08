import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Badge } from "../../../components/ui/Badge";
import { QualityScoreBadge } from "./QualityScoreBadge";

export function ReviewItemCard({ item, onSelect, selected }: any) {
  return (
    <TerminalCard
      className={`p-3 cursor-pointer ${selected ? "border-neon-blue" : ""}`}
      onClick={onSelect}
    >
      <div className="flex justify-between">
        <span className="text-sm text-white">{item.item.id.slice(0, 8)}</span>
        <Badge
          variant={
            item.priority === "high"
              ? "failed"
              : item.priority === "medium"
                ? "warning"
                : "secondary"
          }
        >
          {item.priority}
        </Badge>
      </div>
      <div className="flex gap-2 mt-2 text-xs">
        <span>Annotations: {item.item.annotations.length}</span>
        <span>Pre‑labels: {item.item.pre_labels?.length || 0}</span>
      </div>
      <div className="flex gap-1 mt-1">
        {item.item.annotations.slice(0, 2).map((a: any) => (
          <QualityScoreBadge key={a.id} score={a.confidence} />
        ))}
      </div>
    </TerminalCard>
  );
}
