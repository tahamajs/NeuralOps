import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Spinner } from "../../../components/ui/Spinner";
import { Button } from "../../../components/ui/Button";

export default function RunArtifactsPage() {
  const { runId } = useParams<{ runId: string }>();
  const { data, isLoading } = useQuery({
    queryKey: ["runArtifacts", runId],
    queryFn: async () => (await api.get(`/runs/${runId}/artifacts/`)).data,
  });

  if (isLoading) return <Spinner />;
  const rawFiles = data?.files || data || [];
  const filesList: { name: string; url: string; size?: string }[] = Array.isArray(rawFiles)
    ? rawFiles.map((f: any) => ({
        name: typeof f === "string" ? f : f.name || f.filename || "file",
        url: f.url || "#",
        size: f.size ? `${f.size} MB` : undefined,
      }))
    : Object.entries(rawFiles).map(([name, val]: any) => ({
        name,
        url: typeof val === "string" ? val : val?.url || "#",
        size: typeof val === "object" && val?.size ? `${val.size} MB` : undefined,
      }));

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">ARTIFACTS</h1>
      <Card>
        {filesList.length === 0 ? (
          <p className="text-slate-500">No artifacts found.</p>
        ) : (
          <div className="space-y-2">
            {filesList.map((file, idx) => (
              <div
                key={idx}
                className="flex justify-between items-center bg-cyber-800 p-3 rounded"
              >
                <div>
                  <span className="text-sm text-white font-mono">{file.name}</span>
                  {file.size && (
                    <span className="text-xs text-slate-400 ml-2">({file.size})</span>
                  )}
                </div>
                <a href={file.url} download target="_blank" rel="noopener noreferrer">
                  <Button size="sm" variant="secondary">
                    Download
                  </Button>
                </a>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
