import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Avatar } from "../../../components/ui/Avatar";
import { Button } from "../../../components/ui/Button";
import { Badge } from "../../../components/ui/Badge";
import { X, UserPlus } from "lucide-react";

export function ResearcherCard({ researcher, onDismiss }: any) {
  return (
    <TerminalCard className="p-4">
      <div className="flex gap-4">
        <Avatar src={researcher.avatar_url} size="lg" />
        <div className="flex-1">
          <div className="flex justify-between items-start">
            <div>
              <h3 className="font-bold text-white">{researcher.username}</h3>
              <div className="text-sm text-slate-400">
                {researcher.institution}
              </div>
            </div>
            <button onClick={onDismiss}>
              <X size={14} className="text-slate-400" />
            </button>
          </div>
          <div className="flex flex-wrap gap-1 mt-2">
            {researcher.expertise.slice(0, 3).map((tag) => (
              <Badge key={tag}>{tag}</Badge>
            ))}
          </div>
          <div className="mt-2 text-xs text-slate-400">
            <span className="text-neon-mint">
              {Math.round(researcher.match_score * 100)}% match
            </span>{" "}
            · {researcher.reason}
          </div>
          <div className="flex gap-2 mt-3">
            <Button size="sm" variant="primary">
              <UserPlus size={12} className="mr-1" /> Follow
            </Button>
            <Button size="sm" variant="ghost">
              View Profile
            </Button>
          </div>
        </div>
      </div>
    </TerminalCard>
  );
}
