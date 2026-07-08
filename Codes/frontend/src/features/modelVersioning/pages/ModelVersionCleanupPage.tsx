import { useParams } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import api from "../../../api/client";
import { Button } from "../../../components/ui/Button";
import { Card } from "../../../components/ui/Card";

export default function ModelVersionCleanupPage() {
  const { runId } = useParams<{ runId: string }>();
  const mutation = useMutation({
    mutationFn: async () =>
      api.post(`/api/model-versioning/runs/${runId}/versions/cleanup/`),
    onSuccess: () => alert("Cleanup started"),
  });
  return (
    <div className="space-y-6 animate-fade-in max-w-xl mx-auto">
      <h1 className="text-2xl font-black text-white">
        VERSION <span className="text-neon-blue">CLEANUP</span>
      </h1>
      <Card>
        <p className="text-slate-400 text-sm mb-4">
          Remove old model versions, keeping only the best K (default 3).
        </p>
        <Button
          variant="danger"
          onClick={() => mutation.mutate()}
          disabled={mutation.isPending}
        >
          {mutation.isPending ? "Cleaning..." : "Start Cleanup"}
        </Button>
      </Card>
    </div>
  );
}
