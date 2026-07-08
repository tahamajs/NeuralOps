import { useParams } from "react-router-dom";
import { useQuery, useMutation } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Spinner } from "../../../components/ui/Spinner";
import { Button } from "../../../components/ui/Button";
import { useState } from "react";

export default function ModelCardPage() {
  const { runId } = useParams<{ runId: string }>();
  const { data, isLoading } = useQuery({
    queryKey: ["modelCard", runId],
    queryFn: async () => (await api.get(`/api/runs/${runId}/model-card/`)).data,
  });

  const [repoId, setRepoId] = useState("");
  const pushMutation = useMutation({
    mutationFn: async () =>
      api.post(`/api/runs/${runId}/model-card/push/`, { repo_id: repoId }),
    onSuccess: () => alert("Pushed to HuggingFace Hub"),
  });

  if (isLoading) return <Spinner />;

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">
        MODEL <span className="text-neon-blue">CARD</span>
      </h1>
      <Card>
        <pre className="text-slate-400 whitespace-pre-wrap text-xs max-h-[600px] overflow-auto">
          {data?.model_card_md || "No card generated."}
        </pre>
        <div className="mt-4 flex gap-2">
          <input
            className="bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-white flex-1"
            placeholder="repo_id (e.g., username/model)"
            value={repoId}
            onChange={(e) => setRepoId(e.target.value)}
          />
          <Button
            size="sm"
            onClick={() => pushMutation.mutate()}
            disabled={pushMutation.isPending || !repoId}
          >
            Push to Hub
          </Button>
        </div>
      </Card>
    </div>
  );
}
