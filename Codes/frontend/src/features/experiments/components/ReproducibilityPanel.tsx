import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { Download, GitCommit, Box, Cpu } from "lucide-react";

interface ReproducibilityPanelProps {
  runId: string;
}

export default function ReproducibilityPanel({
  runId,
}: ReproducibilityPanelProps) {
  const { data: snapshot, isLoading } = useQuery({
    queryKey: ["reproducibility", runId],
    queryFn: async () =>
      (await api.get(`/runs/${runId}/reproducibility/`)).data,
    enabled: !!runId,
  });

  if (isLoading) return <Spinner className="py-4" />;
  if (!snapshot) return null;

  return (
    <Card>
      <h3 className="text-sm font-display font-bold text-white mb-4 flex items-center gap-2">
        <Box size={18} className="text-neon-blue" />
        Reproducibility Snapshot
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
        <div className="flex items-center gap-2">
          <GitCommit size={14} className="text-slate-400" />
          <div>
            <span className="text-slate-400">Git Commit</span>
            <div className="text-white font-mono">
              {snapshot.git_commit?.slice(0, 7) || "—"}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Cpu size={14} className="text-slate-400" />
          <div>
            <span className="text-slate-400">Python</span>
            <div className="text-white">{snapshot.python_version || "—"}</div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-slate-400">Seed</span>
          <div className="text-white font-mono">{snapshot.seed ?? "—"}</div>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-slate-400">DVC Hash</span>
          <div className="text-white font-mono">
            {snapshot.dvc_hash?.slice(0, 12) || "—"}
          </div>
        </div>
      </div>
      <Button
        size="sm"
        variant="ghost"
        className="mt-4"
        onClick={() =>
          window.open(`/runs/${runId}/reproducibility/download/`, "_blank")
        }
      >
        <Download size={14} className="mr-1" />
        Download reproduce.py
      </Button>
    </Card>
  );
}
