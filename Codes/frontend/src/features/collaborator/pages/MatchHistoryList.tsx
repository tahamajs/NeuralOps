import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { collaboratorAPI } from "../services/api";
import { MatchHistoryItem } from "../components/MatchHistoryItem";
import { MatchChart } from "../components/MatchChart";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { Clock, BarChart3 } from "lucide-react";

export default function MatchHistoryList({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [statusFilter, setStatusFilter] = useState("");
  const [page, setPage] = useState(1);
  const [viewMode, setViewMode] = useState<"list" | "chart">("list");

  const { data, isLoading } = useQuery({
    queryKey: ["match-history", { status: statusFilter, page }],
    queryFn: () =>
      collaboratorAPI.getMatchHistory({
        status: statusFilter || undefined,
        page,
      }),
  });

  const matches = (data as any)?.results || [];
  const totalPages = Math.ceil(((data as any)?.count || 0) / 20);

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Clock size={20} className="text-neon-mint" /> Collaboration History
          </h1>
          <p className="text-xs text-slate-400">
            Track your past and ongoing collaborations
          </p>
        </div>
        <div className="flex gap-2">
          <Select
            options={[
              { value: "", label: "All" },
              { value: "pending", label: "Pending" },
              { value: "accepted", label: "Accepted" },
              { value: "connected", label: "Connected" },
            ]}
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="w-32"
          />
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setViewMode(viewMode === "list" ? "chart" : "list")}
          >
            {viewMode === "list" ? (
              <BarChart3 size={14} className="mr-1" />
            ) : (
              <Clock size={14} className="mr-1" />
            )}
            {viewMode === "list" ? "Chart" : "List"}
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        {isLoading ? (
          <Spinner />
        ) : viewMode === "chart" ? (
          <MatchChart matches={matches} />
        ) : matches.length === 0 ? (
          <div className="text-center text-slate-500 py-12">
            No collaboration history yet. Accept a suggestion to get started!
          </div>
        ) : (
          matches.map((match) => (
            <MatchHistoryItem key={match.id} match={match} />
          ))
        )}
        {totalPages > 1 && (
          <div className="flex justify-center gap-2 pt-4">
            <Button
              size="sm"
              variant="ghost"
              disabled={page === 1}
              onClick={() => setPage((p) => p - 1)}
            >
              Previous
            </Button>
            <span className="text-sm text-slate-400">
              Page {page} of {totalPages}
            </span>
            <Button
              size="sm"
              variant="ghost"
              disabled={page === totalPages}
              onClick={() => setPage((p) => p + 1)}
            >
              Next
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
