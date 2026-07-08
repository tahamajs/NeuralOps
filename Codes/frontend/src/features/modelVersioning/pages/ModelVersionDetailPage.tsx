import { useParams } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";

export default function ModelVersionDetailPage() {
  const { versionId } = useParams<{ versionId: string }>();
  const queryClient = useQueryClient();

  const { data: version, isLoading } = useQuery({
    queryKey: ["modelVersion", versionId],
    queryFn: async () =>
      (await api.get(`/api/model-versioning/versions/${versionId}/`)).data,
  });

  const publishMutation = useMutation({
    mutationFn: () =>
      api.post(`/api/model-versioning/versions/${versionId}/publish/`),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["modelVersion", versionId] }),
  });
  const archiveMutation = useMutation({
    mutationFn: () =>
      api.post(`/api/model-versioning/versions/${versionId}/archive/`),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["modelVersion", versionId] }),
  });
  const downloadMutation = useMutation({
    mutationFn: () =>
      api.get(`/api/model-versioning/versions/${versionId}/download/`, {
        responseType: "blob",
      }),
    onSuccess: (res) => {
      const url = window.URL.createObjectURL(res.data);
      const a = document.createElement("a");
      a.href = url;
      a.download = `model_v${version.version_number}.pth`;
      a.click();
      window.URL.revokeObjectURL(url);
    },
  });

  if (isLoading) return <Spinner />;

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">
        VERSION <span className="text-neon-blue">{version.version}</span>
      </h1>
      <Card>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-slate-500">Version Number:</span>{" "}
            {version.version_number}
          </div>
          <div>
            <span className="text-slate-500">Size:</span>{" "}
            {version.artifact_size_mb} MB
          </div>
          <div>
            <span className="text-slate-500">Published:</span>{" "}
            {version.is_published_to_hub ? <Badge>Yes</Badge> : "No"}
          </div>
          <div>
            <span className="text-slate-500">Created:</span>{" "}
            {new Date(version.created_at).toLocaleString()}
          </div>
        </div>
        <div className="mt-4 flex gap-2">
          <Button
            size="sm"
            onClick={() => publishMutation.mutate()}
            disabled={publishMutation.isPending}
          >
            Publish to Hub
          </Button>
          <Button
            size="sm"
            variant="secondary"
            onClick={() => archiveMutation.mutate()}
            disabled={archiveMutation.isPending}
          >
            Archive
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => downloadMutation.mutate()}
          >
            Download
          </Button>
        </div>
        {version.metrics && (
          <div className="mt-4">
            <h3 className="text-sm font-bold text-white">Metrics</h3>
            <pre className="bg-cyber-800 p-3 rounded-lg text-xs text-slate-400 mt-1">
              {JSON.stringify(version.metrics, null, 2)}
            </pre>
          </div>
        )}
      </Card>
    </div>
  );
}
