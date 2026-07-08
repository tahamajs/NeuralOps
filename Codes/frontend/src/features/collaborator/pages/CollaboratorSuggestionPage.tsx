import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { collaboratorAPI } from "../services/api";
import { SuggestionCard } from "../components/SuggestionCard";
import { FilterBar } from "../components/FilterBar";
import { EmptyState } from "../components/EmptyState";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { Users, RefreshCw, Filter } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function CollaboratorSuggestionPage({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [filters, setFilters] = useState<MatchFilters>({});
  const [page, setPage] = useState(1);
  const [showFilters, setShowFilters] = useState(false);
  const queryClient = useQueryClient();
  const { success, error: toastError } = useToast();

  const { data, isLoading, refetch } = useQuery({
    queryKey: ["collaborator-suggestions", filters, page],
    queryFn: () => collaboratorAPI.getSuggestions({ ...filters, page }),
  });

  const refreshMutation = useMutation({
    mutationFn: () => collaboratorAPI.refreshSuggestions(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["collaborator-suggestions"] });
      success(
        "Suggestions refreshed",
        "New recommendations are being generated.",
      );
    },
    onError: () => toastError("Failed to refresh suggestions"),
  });

  const suggestions = data?.results || [];
  const totalPages = Math.ceil((data?.count || 0) / 20);

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Users size={20} className="text-neon-mint" /> Collaborator
            Suggestions
          </h1>
          <p className="text-xs text-slate-400">
            AI‑powered recommendations based on your research profile
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setShowFilters(!showFilters)}
          >
            <Filter size={14} className="mr-1" /> Filter
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => refreshMutation.mutate()}
          >
            <RefreshCw size={14} className="mr-1" /> Refresh
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        {showFilters && (
          <FilterBar
            filters={filters}
            onFilterChange={setFilters}
            onApply={() => setPage(1)}
          />
        )}
        {isLoading ? (
          <Spinner />
        ) : suggestions.length === 0 ? (
          <EmptyState type="suggestions" />
        ) : (
          <div className="grid md:grid-cols-2 gap-4">
            {suggestions.map((suggestion) => (
              <SuggestionCard
                key={suggestion.id}
                suggestion={suggestion}
                onAccept={() => {
                  queryClient.invalidateQueries({
                    queryKey: ["collaborator-suggestions"],
                  });
                  success("Collaboration request sent");
                }}
                onReject={() => {
                  queryClient.invalidateQueries({
                    queryKey: ["collaborator-suggestions"],
                  });
                  success("Suggestion dismissed");
                }}
              />
            ))}
          </div>
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
