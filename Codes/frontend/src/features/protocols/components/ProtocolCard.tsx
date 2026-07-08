import { Link } from "react-router-dom";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Badge } from "../../../components/ui/Badge";
import { Button } from "../../../components/ui/Button";
import { Calendar, GitBranch, Trash2, Eye } from "lucide-react";

export function ProtocolCard({ protocol, onDelete }: any) {
  return (
    <TerminalCard className="p-4">
      <div className="flex justify-between items-start">
        <div>
          <Link to={`/protocols/${protocol.id}`}>
            <h3 className="font-bold text-white hover:text-neon-mint transition-colors">
              {protocol.name}
            </h3>
          </Link>
          <p className="text-sm text-slate-400 mt-1 line-clamp-2">
            {protocol.description}
          </p>
          <div className="flex flex-wrap gap-1 mt-2">
            {protocol.tags.map((t) => (
              <Badge key={t} variant="info">
                {t}
              </Badge>
            ))}
          </div>
          <div className="flex items-center gap-2 mt-2 text-xs text-slate-500">
            <span className="flex items-center gap-1">
              <GitBranch size={12} /> v{protocol.version}
            </span>
            <span className="flex items-center gap-1">
              <Calendar size={12} />{" "}
              {new Date(protocol.updated_at).toLocaleDateString()}
            </span>
            <Badge
              variant={
                protocol.status === "published"
                  ? "success"
                  : protocol.status === "draft"
                    ? "warning"
                    : "secondary"
              }
            >
              {protocol.status}
            </Badge>
          </div>
        </div>
        <div className="flex gap-1">
          <Link to={`/protocols/${protocol.id}`}>
            <Button size="sm" variant="ghost">
              <Eye size={14} />
            </Button>
          </Link>
          <button onClick={() => onDelete(protocol.id)}>
            <Trash2 size={14} className="text-red-400" />
          </button>
        </div>
      </div>
    </TerminalCard>
  );
}
