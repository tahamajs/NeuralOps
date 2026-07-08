import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Badge } from "../../../components/ui/Badge";
import { StageBadge } from "./StageBadge";
import { Model } from "../types";
import { Link } from "react-router-dom";

export function ModelCard({ model }: { model: Model }) {
  return (
    <Link to={`/model-hub/models/${model.id}`}>
      <TerminalCard className="p-4">
        <div className="flex justify-between items-start">
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-bold text-white">{model.name}</h3>
              <Badge variant="secondary">v{model.version}</Badge>
              <StageBadge stage={model.stage} />
            </div>
            <p className="text-sm text-slate-400 mt-1 line-clamp-2">
              {model.description || "No description"}
            </p>
            <div className="flex flex-wrap gap-2 mt-2">
              <Badge variant="info">{model.framework}</Badge>
              <Badge variant="info">{model.task}</Badge>
              {model.tags.slice(0, 2).map((t) => (
                <Badge key={t}>{t}</Badge>
              ))}
            </div>
            <div className="text-xs text-slate-500 mt-2">
              Created: {new Date(model.created_at).toLocaleDateString()} by{" "}
              {model.created_by}
            </div>
          </div>
          <div className="text-right">
            <div className="text-xs text-slate-400">Metrics</div>
            {Object.entries(model.metrics)
              .slice(0, 2)
              .map(([k, v]) => (
                <div key={k} className="text-sm text-white">
                  {k}: {v.toFixed(4)}
                </div>
              ))}
          </div>
        </div>
      </TerminalCard>
    </Link>
  );
}
