import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Badge } from "../../../components/ui/Badge";
import { Database, Upload, Download } from "lucide-react";
import { FeatureGroup } from "../types";

export function FeatureGroupCard({
  group,
  onRefresh,
}: {
  group: FeatureGroup;
  onRefresh: () => void;
}) {
  return (
    <TerminalCard className="p-4">
      <div className="flex justify-between items-start">
        <div>
          <div className="flex items-center gap-2">
            <Database size={16} className="text-neon-mint" />
            <h3 className="font-bold text-white">{group.name}</h3>
          </div>
          <div className="text-xs text-slate-400 mt-1">
            Entity: {group.entity_key}
          </div>
          <div className="flex flex-wrap gap-1 mt-2">
            {group.features.slice(0, 5).map((f) => (
              <Badge key={f.name} variant="info">
                {f.name}
              </Badge>
            ))}
          </div>
        </div>
        <div className="flex gap-1">
          <Button size="sm" variant="ghost" title="Ingest">
            <Upload size={14} />
          </Button>
          <Button size="sm" variant="ghost" title="Serve">
            <Download size={14} />
          </Button>
        </div>
      </div>
    </TerminalCard>
  );
}
