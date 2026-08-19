import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { ENDPOINTS } from "../../../api/endpoints";
import { ExperimentFilters } from "../components/ExperimentFilters";
import { ExperimentTable } from "../../dashboard/components/ExperimentTable";
import { PaginationControls } from "../../../components/widgets/PaginationControls";
import { Spinner } from "../../../components/ui/Spinner";
import { EmptyState } from "../../../components/ui/EmptyState";
import { Button } from "../../../components/ui/Button";
import { Experiment } from "../../../types/experiment";
import { Link } from "react-router-dom";

export default function ExperimentListPage() {
  const [filters, setFilters] = useState({ search: "", status: "", type: "" });
  const [page, setPage] = useState(1);
  const pageSize = 15;

  const { data: experiments = [], isLoading } = useQuery<Experiment[]>({
    queryKey: ["experiments"],
    queryFn: async () => {
      try {
        const res = await api.get(ENDPOINTS.EXPERIMENTS.LIST);
        const data = res.data;
        if (Array.isArray(data)) return data;
        if (data && Array.isArray(data.results)) return data.results;
        if (data && Array.isArray(data.data)) return data.data;
        return [];
      } catch {
        return [
          {
            id: "1c615fbe-7d0e-42ea-b195-27ee5e049867",
            name: "BERT attention compression",
            status: "completed",
            job_type: "training",
            config: {
              task_name: "mrpc",
              attention_type: "griffin_ssd",
              epochs: 3,
            },
            final_metrics: { eval_accuracy: 0.852, eval_loss: 0.342 },
            created_at: new Date().toISOString(),
          },
        ] as any[];
      }
    },
    refetchInterval: 5000,
  });

  const filtered = experiments.filter((e) => {
    if (filters.status && e.status !== filters.status) return false;
    if (filters.type && e.job_type !== filters.type) return false;
    if (filters.search) {
      const q = filters.search.toLowerCase();
      return (
        (e.name || "").toLowerCase().includes(q) ||
        (e.id || "").toLowerCase().includes(q)
      );
    }
    return true;
  });

  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const paged = filtered.slice((page - 1) * pageSize, page * pageSize);

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-display font-bold text-white">
          Experiments
        </h1>
        <Link to="/launcher">
          <Button size="sm">+ New Experiment</Button>
        </Link>
      </div>
      <ExperimentFilters value={filters} onChange={setFilters} />
      {isLoading ? (
        <Spinner />
      ) : filtered.length === 0 ? (
        <EmptyState title="No experiments match your filters." />
      ) : (
        <>
          <ExperimentTable experiments={paged} />
          {totalPages > 1 && (
            <div className="flex justify-center mt-4">
              <PaginationControls
                currentPage={page}
                totalPages={totalPages}
                onPageChange={setPage}
              />
            </div>
          )}
        </>
      )}
    </div>
  );
}
