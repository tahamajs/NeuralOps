import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Badge } from "../../../components/ui/Badge";
import { Spinner } from "../../../components/ui/Spinner";
import { useNavigate } from "react-router-dom";

export default function ModelRegistryPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [filter, setFilter] = useState("");

  const { data: models = [], isLoading } = useQuery({
    queryKey: ["modelVersions"],
    queryFn: async () =>
      (await api.get("/api/model-versioning/versions/")).data,
    refetchInterval: 10000,
  });

  const promoteMut = useMutation({
    mutationFn: (id: string) =>
      api.post(`/api/model-versioning/versions/${id}/promote/`),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["modelVersions"] }),
  });

  const archiveMut = useMutation({
    mutationFn: (id: string) =>
      api.post(`/api/model-versioning/versions/${id}/archive/`),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["modelVersions"] }),
  });

  const filtered = models.filter(
    (m: any) =>
      m.name?.toLowerCase().includes(filter.toLowerCase()) ||
      m.version?.toLowerCase().includes(filter.toLowerCase()),
  );

  if (isLoading) return <Spinner />;

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-display font-bold text-white">
          Model Registry
        </h1>
        <Input
          placeholder="Search models..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="w-64"
        />
      </div>

      <div className="grid gap-4">
        {filtered.map((model: any) => (
          <Card key={model.id} className="flex items-center justify-between">
            <div
              className="flex-1 cursor-pointer"
              onClick={() => navigate(`/model-versions/${model.id}`)}
            >
              <div className="flex items-center gap-3">
                <h3 className="text-sm font-bold text-white">
                  {model.name}{" "}
                  <span className="text-slate-400 text-xs">
                    v{model.version_number}
                  </span>
                </h3>
                <Badge status={model.deployment_stage || "none"}>
                  {model.deployment_stage || "none"}
                </Badge>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Created {new Date(model.created_at).toLocaleDateString()} ·
                Accuracy: {model.metrics?.eval_accuracy?.toFixed(4) || "—"}
              </p>
            </div>
            <div className="flex gap-2 ml-4">
              <Button
                size="sm"
                variant="ghost"
                onClick={() => promoteMut.mutate(model.id)}
              >
                Promote
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => archiveMut.mutate(model.id)}
              >
                Archive
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() =>
                  window.open(
                    `/api/model-versioning/versions/${model.id}/download/`,
                    "_blank",
                  )
                }
              >
                Download
              </Button>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
