import { Link } from "react-router-dom";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Badge } from "../../../components/ui/Badge";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import type { ResearchLandscape } from "../types";

export function LandscapeCard({ landscape }: { landscape: ResearchLandscape }) {
  const trendIcon = {
    rising: <TrendingUp size={14} className="text-green-400" />,
    stable: <Minus size={14} className="text-yellow-400" />,
    declining: <TrendingDown size={14} className="text-red-400" />,
  }[landscape.trend];
  return (
    <Link to={`/literature/landscapes/${landscape.id}`}>
      <TerminalCard className="p-4">
        <div className="flex justify-between items-start">
          <div>
            <h3 className="font-bold text-white">{landscape.name}</h3>
            <div className="flex flex-wrap gap-1 mt-1">
              {landscape.topics.slice(0, 3).map((t) => (
                <Badge key={t}>{t}</Badge>
              ))}
            </div>
          </div>
          {trendIcon}
        </div>
        <div className="mt-2 text-xs text-slate-400">
          {landscape.paper_count} papers
        </div>
      </TerminalCard>
    </Link>
  );
}
