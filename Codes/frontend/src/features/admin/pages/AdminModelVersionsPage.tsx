import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { Box } from "lucide-react";

export default function AdminModelVersionsPage() {
  const { data: experiments = [], isLoading } = useQuery({
    queryKey: ["experiments"],
    queryFn: async () => (await api.get("/experiments/")).data,
  });

  if (isLoading) return <Spinner />;

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-display font-bold text-white flex items-center gap-2">
        <Box size={24} className="text-neon-blue" /> Model Versions
      </h1>
      <p className="text-slate-400 text-sm">
        Select an experiment to manage its model versions.
      </p>
      <div className="grid gap-4">
        {experiments.length === 0 && (
          <p className="text-slate-500 text-sm text-center py-8">
            No experiments found.
          </p>
        )}
        {experiments.map((exp: any) => (
          <Card
            key={exp.id}
            hover
            className="flex items-center justify-between"
          >
            <div>
              <span className="text-white font-semibold">
                {exp.name || exp.id.slice(0, 8)}
              </span>
              <p className="text-xs text-slate-400 mt-0.5">
                {exp.job_type} – {exp.status}
              </p>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() =>
                window.open(`/runs/${exp.id}/model-versions`, "_blank")
              }
            >
              Manage Versions
            </Button>
          </Card>
        ))}
      </div>
    </div>
  );
}
