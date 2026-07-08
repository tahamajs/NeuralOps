import { useParams, Link } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import api from "../../../api/client";
import { Button } from "../../../components/ui/Button";
import { Card } from "../../../components/ui/Card";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { EmptyState } from "../../../components/ui/EmptyState";

export default function ModelVersionsPage() {
  const { runId } = useParams<{ runId: string }>();
  const queryClient = useQueryClient();

  const { data: versions = [], isLoading } = useQuery({
    queryKey: ["modelVersions", runId],
    queryFn: async () =>
      (await api.get(`/api/model-versioning/runs/${runId}/versions/`)).data,
  });

  const publishMutation = useMutation({
    mutationFn: async (versionId: string) =>
      api.post(`/api/model-versioning/versions/${versionId}/publish/`),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["modelVersions", runId] }),
  });

  if (isLoading) return <Spinner />;
  if (!versions.length) return <EmptyState message="No model versions yet" />;

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">Model Versions</h1>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {versions.map((v: any) => (
          <Card key={v.id} hover>
            <div className="flex items-center justify-between mb-2">
              <Badge status="completed">{v.version}</Badge>
              {v.is_published_to_hub && (
                <span className="text-green-400 text-xs">Published</span>
              )}
            </div>
            <div className="text-xs text-slate-400 space-y-1 mb-3">
              <p>Size: {v.artifact_size_mb} MB</p>
              <p>Metrics: {JSON.stringify(v.metrics)}</p>
            </div>
            <div className="flex gap-2">
              <Button
                size="sm"
                variant="secondary"
                onClick={() => publishMutation.mutate(v.id)}
              >
                Publish to Hub
              </Button>
              <a
                href={`/api/model-versioning/versions/${v.id}/download/`}
                download
              >
                <Button size="sm" variant="ghost">
                  Download
                </Button>
              </a>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
