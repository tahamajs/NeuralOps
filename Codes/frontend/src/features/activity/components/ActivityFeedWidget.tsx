import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Spinner } from "../../../components/ui/Spinner";
import { formatDistanceToNow } from "date-fns";
import {
  GitBranch,
  FileText,
  StickyNote,
  MessageSquare,
  Plus,
} from "lucide-react";

export function ActivityFeedWidget() {
  const { data, isLoading } = useQuery({
    queryKey: ["activity-feed"],
    queryFn: async () => (await api.get("/activity/?limit=10")).data,
    refetchInterval: 30000,
  });

  if (isLoading) return <Spinner size="sm" />;

  const getIcon = (type: string) => {
    switch (type) {
      case "experiment":
        return <GitBranch size={14} className="text-neon-blue" />;
      case "note":
        return <StickyNote size={14} className="text-yellow-400" />;
      case "paper":
        return <FileText size={14} className="text-purple-400" />;
      case "question":
        return <MessageSquare size={14} className="text-green-400" />;
      default:
        return <Plus size={14} className="text-slate-400" />;
    }
  };

  return (
    <Card className="p-4">
      <h3 className="text-sm font-bold text-white mb-3">Recent Activity</h3>
      <div className="space-y-3">
        {data?.results?.map((activity: any) => (
          <div key={activity.id} className="flex items-start gap-3">
            {getIcon(activity.target_type)}
            <div className="flex-1">
              <p className="text-sm text-slate-300">{activity.action}</p>
              <p className="text-xs text-slate-500">
                {formatDistanceToNow(new Date(activity.created_at), {
                  addSuffix: true,
                })}
              </p>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}
