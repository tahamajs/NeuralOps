// src/features/modelHub/pages/ModelRegistryList.tsx
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { modelHubAPI } from "../services/api";
import { ModelCard } from "../components/ModelCard";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { Database, Filter, Plus } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function ModelRegistryList({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [search, setSearch] = useState("");
  const [framework, setFramework] = useState("");
  const [task, setTask] = useState("");
  const [stage, setStage] = useState("");
  const [page, setPage] = useState(1);
  const { success, error } = useToast();

  const { data, isLoading, refetch } = useQuery({
    queryKey: ["models", { search, framework, task, stage, page }],
    queryFn: () =>
      modelHubAPI.getModels({
        q: search,
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
            <Database size={20} className="text-neon-mint" /> Model Registry
          </h1>
          <p className="text-xs text-slate-400">
            Central model versioning and lifecycle management
          </p>
        </div>
        <div className="flex gap-2">
          <Button size="sm">
            <Plus size={14} className="mr-1" /> Register Model
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <div className="flex flex-wrap gap-3 items-end">
          <Input
            placeholder="Search by name, tags..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-64"
          />
          <Select
            options={[
              { value: "", label: "All Frameworks" },
              { value: "pytorch", label: "PyTorch" },
              { value: "tensorflow", label: "TensorFlow" },
              { value: "onnx", label: "ONNX" },
            ]}
            value={framework}
            onChange={(e) => setFramework(e.target.value)}
            className="w-40"
          />
          <Select
            options={[
              { value: "", label: "All Tasks" },
              { value: "classification", label: "Classification" },
              { value: "detection", label: "Detection" },
              { value: "segmentation", label: "Segmentation" },
            ]}
            value={task}
            onChange={(e) => setTask(e.target.value)}
            className="w-40"
          />
          <Select
            options={[
              { value: "", label: "All Stages" },
              { value: "development", label: "Development" },
              { value: "staging", label: "Staging" },
              { value: "production", label: "Production" },
            ]}
            value={stage}
            onChange={(e) => setStage(e.target.value)}
            className="w-36"
          />
          <Button size="sm" onClick={() => refetch()}>
            Apply
          </Button>
        </div>
        {isLoading ? (
          <Spinner />
        ) : (
          <div className="grid md:grid-cols-2 gap-4">
            {models.map((m) => (
              <ModelCard key={m.id} model={m} />
            ))}
          </div>
        )}
        {totalPages > 1 && (
          <div className="flex justify-center gap-2 mt-4">
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
