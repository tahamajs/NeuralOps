import { Link } from "react-router-dom";
import { Card, Badge, Button, Avatar } from "../../../components/ui";
import { formatDistanceToNow } from "date-fns";
import {
  AlertCircle,
  Clock,
  Star,
  Users,
  Award,
  GitBranch,
  FileText,
  FlaskConical,
} from "lucide-react";
import { cn } from "../../../../lib/utils";

interface ReviewRequestCardProps {
  request: ReviewRequest;
  variant?: "dashboard" | "compact" | "full";
  onAccept?: (id: string) => void;
  onDecline?: (id: string) => void;
}

export function ReviewRequestCard({
  request,
  variant = "full",
  onAccept,
  onDecline,
}: ReviewRequestCardProps) {
  const isExpired = new Date(request.due_date) < new Date();
  const priorityColors = {
    high: "border-red-500/30 bg-red-500/5",
    medium: "border-yellow-500/30 bg-yellow-500/5",
    low: "border-green-500/30 bg-green-500/5",
  };

  const getContextIcon = () => {
    if (request.context.experiment_id)
      return <FlaskConical size={14} className="text-purple-400" />;
    if (request.context.run_id)
      return <GitBranch size={14} className="text-green-400" />;
    if (request.context.paper_id)
      return <FileText size={14} className="text-blue-400" />;
    return null;
  };

  if (variant === "compact") {
    return (
      <Link to={`/reviews/${request.id}`}>
        <Card hover className="p-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {getContextIcon()}
              <div>
                <h4 className="text-sm font-medium text-white">
                  {request.title}
                </h4>
                <div className="flex items-center gap-2 text-xs text-slate-400">
                  <span>by {request.requester.username}</span>
                  <span>•</span>
                  <span>
                    {formatDistanceToNow(new Date(request.created_at), {
                      addSuffix: true,
                    })}
                  </span>
                </div>
              </div>
            </div>
            <Badge
              variant={
                request.status === "pending"
                  ? "warning"
                  : request.status === "in_progress"
                    ? "info"
                    : "success"
              }
            >
              {request.status}
            </Badge>
          </div>
        </Card>
      </Link>
    );
  }

  return (
    <Card
      className={cn(
        "p-5 transition-all duration-200 hover:scale-[1.01]",
        priorityColors[request.priority],
      )}
    >
      <div className="flex flex-wrap justify-between items-start gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-2">
            {getContextIcon()}
            <Link to={`/reviews/${request.id}`}>
              <h3 className="text-lg font-semibold text-white hover:text-neon-blue transition-colors">
                {request.title}
              </h3>
            </Link>
          </div>

          <p className="text-sm text-slate-400 mb-3 line-clamp-2">
            {request.description}
          </p>

          <div className="flex flex-wrap items-center gap-3 text-xs">
            <div className="flex items-center gap-1 text-slate-500">
              <Avatar src={request.requester.avatar_url} size="xs" />
              <span>{request.requester.username}</span>
            </div>

            {request.reviewer && (
              <div className="flex items-center gap-1 text-slate-500">
                <Users size={12} />
                <span>Assigned to {request.reviewer.username}</span>
              </div>
            )}

            <div
              className={cn(
                "flex items-center gap-1",
                isExpired ? "text-red-400" : "text-slate-500",
              )}
            >
              <Clock size={12} />
              <span>
                Due{" "}
                {formatDistanceToNow(new Date(request.due_date), {
                  addSuffix: true,
                })}
              </span>
            </div>

            {request.bounty > 0 && (
              <div className="flex items-center gap-1 text-yellow-400">
                <Award size={12} />
                <span>{request.bounty} tokens</span>
              </div>
            )}
          </div>

          <div className="flex flex-wrap gap-1 mt-3">
            {request.tags?.map((tag) => (
              <span
                key={tag}
                className="px-2 py-0.5 rounded bg-white/5 text-slate-400 text-[10px]"
              >
                #{tag}
              </span>
            ))}
          </div>
        </div>

        <div className="flex flex-col items-end gap-2">
          <Badge
            variant={
              request.status === "pending"
                ? "warning"
                : request.status === "in_progress"
                  ? "info"
                  : "success"
            }
          >
            {request.status === "pending"
              ? "Pending"
              : request.status === "in_progress"
                ? "In Progress"
                : "Completed"}
          </Badge>

          {request.status === "pending" && onAccept && onDecline && (
            <div className="flex gap-2 mt-2">
              <Button
                size="sm"
                variant="primary"
                onClick={() => onAccept(request.id)}
              >
                Accept
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => onDecline(request.id)}
              >
                Decline
              </Button>
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}
