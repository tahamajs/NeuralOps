import { useState } from "react";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Badge } from "../../../components/ui/Badge";
import { Avatar } from "../../../components/ui/Avatar";
import {
  Star,
  StarHalf,
  MessageCircle,
  Calendar,
  Award,
  TrendingUp,
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import { MatchHistoryEntry } from "../types";

export function MatchHistoryItem({ match }: { match: MatchHistoryEntry }) {
  const [showFeedback, setShowFeedback] = useState(false);
  const [rating, setRating] = useState(0);

  const statusConfig = {
    pending: { label: "Pending", color: "warning" },
    accepted: { label: "Accepted", color: "info" },
    rejected: { label: "Rejected", color: "danger" },
    connected: { label: "Connected", color: "success" },
  };

  return (
    <TerminalCard className="p-4">
      <div className="flex gap-4">
        <Avatar src={match.collaborator.avatar_url} size="md" />
        <div className="flex-1">
          <div className="flex justify-between items-start">
            <div>
              <h3 className="font-bold text-white">
                {match.collaborator.name}
              </h3>
              <div className="text-xs text-slate-400">
                {match.collaborator.institution}
              </div>
              <div className="flex gap-2 mt-1">
                <Badge variant={statusConfig[match.match_status].color as any}>
                  {statusConfig[match.match_status].label}
                </Badge>
                <Badge variant="secondary">{match.collaboration_type}</Badge>
              </div>
            </div>
            <div className="text-right text-sm">
              <div className="text-slate-400">
                {formatDistanceToNow(new Date(match.created_at), {
                  addSuffix: true,
                })}
              </div>
              {match.messages_exchanged > 0 && (
                <div className="text-xs text-neon-mint">
                  💬 {match.messages_exchanged} messages
                </div>
              )}
            </div>
          </div>
          {match.project_name && (
            <div className="mt-2 text-sm">
              <span className="text-slate-400">Project:</span>{" "}
              <span className="text-white">{match.project_name}</span>
            </div>
          )}
          {match.outcome && (
            <div className="mt-1 text-sm text-green-400">✓ {match.outcome}</div>
          )}
          {match.rating && (
            <div className="mt-1 flex items-center gap-1">
              {Array.from({ length: 5 }).map((_, i) =>
                i < match.rating! ? (
                  <Star
                    key={i}
                    size={14}
                    className="fill-yellow-400 text-yellow-400"
                  />
                ) : (
                  <Star key={i} size={14} className="text-slate-500" />
                ),
              )}
            </div>
          )}
          {!match.rating &&
            match.match_status === "connected" &&
            !showFeedback && (
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setShowFeedback(true)}
                className="mt-2"
              >
                Rate this collaboration
              </Button>
            )}
          {showFeedback && (
            <div className="mt-3 pt-2 border-t border-white/10">
              <div className="text-xs text-slate-400 mb-1">
                How was your experience?
              </div>
              <div className="flex gap-1">
                {Array.from({ length: 5 }).map((i) => (
                  <button
                    key={i}
                    onClick={() => setRating(i + 1)}
                    className="focus:outline-none"
                  >
                    {i < rating ? (
                      <Star
                        size={20}
                        className="fill-yellow-400 text-yellow-400"
                      />
                    ) : (
                      <Star size={20} className="text-slate-500" />
                    )}
                  </button>
                ))}
              </div>
              <div className="flex gap-2 mt-2">
                <Button
                  size="sm"
                  onClick={() => {
                    /* submit rating */ setShowFeedback(false);
                  }}
                >
                  Submit
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => setShowFeedback(false)}
                >
                  Cancel
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>
    </TerminalCard>
  );
}
