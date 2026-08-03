import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { expertWallAPI } from "../services/api";
import { ExpertCard } from "../components/ExpertCard";
import { FilterBar } from "../components/FilterBar";
import { StatsWidget } from "../components/StatsWidget";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Spinner } from "../../../components/ui/Spinner";
import {
  Search,
  Users,
  Filter,
  Grid3X3,
  LayoutList,
  RefreshCw,
} from "lucide-react";
import { useToast } from "../../../components/ui/Toast";
import type { FilterOptions } from "../types";

export default function ExpertDirectoryPage({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [filters, setFilters] = useState<FilterOptions>({});
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [searchTerm, setSearchTerm] = useState("");
  const [showFilters, setShowFilters] = useState(false);
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const {
    data: expertsData,
    isLoading,
    refetch,
  } = useQuery({
    queryKey: ["experts", filters, searchTerm],
    queryFn: () =>
      (expertWallAPI as any).getExperts({
        ...filters,
        search: searchTerm || undefined,
      }),
  });

  const { data: stats } = useQuery({
    queryKey: ["expert-stats"],
    queryFn: () => (expertWallAPI as any).getStats(),
  });

  const refreshMutation = useMutation({
    mutationFn: () => (expertWallAPI as any).refreshExpertCache(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["experts"] });
      success("Expert list refreshed");
    },
    onError: () => error("Failed to refresh"),
  });

  const experts = expertsData?.results || [];
  const totalPages = Math.ceil((expertsData?.count || 0) / 20);

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Users size={20} className="text-neon-mint" /> Expert Directory
          </h1>
          <p className="text-xs text-slate-400">
            Find and connect with domain experts
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
            onClick={() => setViewMode(viewMode === "grid" ? "list" : "grid")}
          >
            {viewMode === "grid" ? (
              <LayoutList size={14} className="mr-1" />
            ) : (
              <Grid3X3 size={14} className="mr-1" />
            )}{" "}
            {viewMode === "grid" ? "List" : "Grid"}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => refreshMutation.mutate()}
          >
            <RefreshCw size={14} /> Refresh
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-auto p-4 space-y-4">
        <StatsWidget stats={stats?.data} />

        <div className="relative max-w-md">
          <Search
            size={16}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"
          />
          <Input
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search by name, expertise, institution..."
            className="pl-9"
          />
        </div>

        {showFilters && (
          <FilterBar filters={filters} onFilterChange={setFilters} />
        )}

        {isLoading ? (
          <Spinner />
        ) : experts.length === 0 ? (
          <TerminalCard className="p-12 text-center">
            <Users
              size={48}
              className="mx-auto mb-4 text-slate-500 opacity-30"
            />
            <h3 className="text-lg font-medium text-white mb-1">
              No experts found
            </h3>
            <p className="text-slate-400 text-sm">
              Try adjusting your search or filters.
            </p>
          </TerminalCard>
        ) : (
          <div
            className={
              viewMode === "grid"
                ? "grid md:grid-cols-2 lg:grid-cols-3 gap-4"
                : "space-y-3"
            }
          >
            {experts.map((expert) => (
              <ExpertCard
                key={expert.id}
                expert={expert}
                variant={viewMode === "grid" ? "card" : "list"}
              />
            ))}
          </div>
        )}

        {totalPages > 1 && (
          <div className="flex justify-center gap-2 pt-4">
            <Button
              size="sm"
              variant="ghost"
              disabled={filters.page === 1}
              onClick={() =>
                setFilters((f) => ({ ...f, page: (f.page || 1) - 1 }))
              }
            >
              Previous
            </Button>
            <span className="text-sm text-slate-400">
              Page {filters.page || 1} of {totalPages}
            </span>
            <Button
              size="sm"
              variant="ghost"
              disabled={filters.page === totalPages}
              onClick={() =>
                setFilters((f) => ({ ...f, page: (f.page || 1) + 1 }))
              }
            >
              Next
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
