// src/features/communityLab/components/DiscoveryFeed.tsx
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { communityLabAPI } from "../services/api";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { Search, Bookmark, Share2 } from "lucide-react";
import { formatDistanceToNow } from "date-fns";

export function DiscoveryFeed({ labId }: { labId: string }) {
  const [filter, setFilter] = useState("all");
  const { data: discoveries, isLoading } = useQuery({
    queryKey: ["discoveries", labId, filter],
    queryFn: () =>
      communityLabAPI.getDiscoveries(labId, {
        type: filter !== "all" ? filter : undefined,
      }),
  });

  if (isLoading) return <Spinner />;

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <Search
          size={16}
          className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"
        />
        <div className="flex gap-1 overflow-x-auto pb-1">
          <button
            onClick={() => setFilter("all")}
            className={`px-3 py-1 rounded text-xs ${filter === "all" ? "bg-neon-blue text-white" : "bg-white/10 text-slate-300"}`}
          >
            All
          </button>
          <button
            onClick={() => setFilter("paper")}
            className={`px-3 py-1 rounded text-xs ${filter === "paper" ? "bg-neon-blue text-white" : "bg-white/10 text-slate-300"}`}
          >
            Papers
          </button>
          <button
            onClick={() => setFilter("dataset")}
            className={`px-3 py-1 rounded text-xs ${filter === "dataset" ? "bg-neon-blue text-white" : "bg-white/10 text-slate-300"}`}
          >
            Datasets
          </button>
          <button
            onClick={() => setFilter("hypothesis")}
            className={`px-3 py-1 rounded text-xs ${filter === "hypothesis" ? "bg-neon-blue text-white" : "bg-white/10 text-slate-300"}`}
          >
            Hypotheses
          </button>
        </div>
      </div>
      {discoveries?.map((item) => (
        <TerminalCard key={item.id} className="p-3">
          <div className="flex justify-between">
            <div>
              <div className="flex items-center gap-2">
                <Badge variant="secondary">{item.type}</Badge>
                <span className="text-xs text-slate-500">
                  {formatDistanceToNow(new Date(item.created_at), {
                    addSuffix: true,
                  })}
                </span>
              </div>
              <h3 className="font-bold text-white mt-1">{item.title}</h3>
              <p className="text-sm text-slate-400 mt-1 line-clamp-2">
                {item.description}
              </p>
              <div className="flex items-center gap-3 mt-2 text-xs text-slate-500">
                <span>By {item.author.username}</span>
                <span>★ {item.score}</span>
              </div>
            </div>
            <div className="flex gap-1">
              <button>
                <Bookmark
                  size={14}
                  className="text-slate-400 hover:text-yellow-400"
                />
              </button>
              <button>
                <Share2
                  size={14}
                  className="text-slate-400 hover:text-neon-blue"
                />
              </button>
            </div>
          </div>
        </TerminalCard>
      ))}
    </div>
  );
}
