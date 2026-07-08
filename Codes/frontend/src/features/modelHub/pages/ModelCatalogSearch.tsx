// src/features/modelHub/pages/ModelCatalogSearch.tsx
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { modelHubAPI } from "../services/api";
import { ModelCard } from "../components/ModelCard";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Input } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { Search, Filter } from "lucide-react";

export default function ModelCatalogSearch({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [query, setQuery] = useState("");
  const [framework, setFramework] = useState("");
  const [task, setTask] = useState("");
  const [stage, setStage] = useState("");
  const [page, setPage] = useState(1);

  const { data, isLoading, refetch } = useQuery({
    queryKey: ["catalog-search", { q: query, framework, task, stage, page }],
    queryFn: () =>
      modelHubAPI.searchCatalog({
        q: query,
        framework: framework || undefined,
        task: task || undefined,
        stage: (stage as any) || undefined,
        page,
      }),
  });

  const models = data?.results || [];
  const totalPages = Math.ceil((data?.count || 0) / 20);

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Search size={20} className="text-neon-mint" /> Model Catalog
          </h1>
          <p className="text-xs text-slate-400">
            Discover and explore registered models
          </p>
        </div>
        {onClose && (
          <Button size="sm" variant="ghost" onClick={onClose}>
            ✕
          </Button>
        )}
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <div className="flex flex-wrap gap-3 items-end">
          <Input
            placeholder="Search by name, tags, description..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="flex-1"
          />
          <Select
            options={[
              { value: "", label: "Framework" },
              { value: "pytorch", label: "PyTorch" },
              { value: "tensorflow", label: "TensorFlow" },
            ]}
            value={framework}
            onChange={(e) => setFramework(e.target.value)}
            className="w-36"
          />
          <Select
            options={[
              { value: "", label: "Task" },
              { value: "classification", label: "Classification" },
              { value: "detection", label: "Detection" },
            ]}
            value={task}
            onChange={(e) => setTask(e.target.value)}
            className="w-36"
          />
          <Select
            options={[
              { value: "", label: "Stage" },
              { value: "development", label: "Development" },
              { value: "staging", label: "Staging" },
              { value: "production", label: "Production" },
            ]}
            value={stage}
            onChange={(e) => setStage(e.target.value)}
            className="w-36"
          />
          <Button onClick={() => refetch()}>Search</Button>
        </div>
        {isLoading ? (
          <Spinner />
        ) : models.length === 0 ? (
          <div className="text-center text-slate-500 py-12">
            No models found
          </div>
        ) : (
          <div className="grid md:grid-cols-2 gap-4">
            {models.map((m) => (
              <ModelCard key={m.id} model={m} />
            ))}
          </div>
        )}
        {totalPages > 1 && (
          <div className="flex justify-center gap-2">
            <Button
              size="sm"
              variant="ghost"
              disabled={page === 1}
              onClick={() => setPage((p) => p - 1)}
            >
              Prev
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
