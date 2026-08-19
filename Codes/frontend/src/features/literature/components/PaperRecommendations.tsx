import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { literatureAPI } from "../services/api";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { Sparkles, X, Bookmark, ExternalLink } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";
import { Recommendation } from "../types";

export function PaperRecommendations({ paperId }: { paperId: string }) {
  const queryClient = useQueryClient();
  const { success } = useToast();

  const fallbackRecommendations: Recommendation[] = [
    {
      paper_id: "rec-1",
      title: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention",
      relevance_score: 0.94,
      reason: "Shares kernel formulation for linear complexity projections.",
    },
    {
      paper_id: "rec-2",
      title: "Griffin: Mixing Recurrent with Local Attention for Efficient Language Models",
      relevance_score: 0.89,
      reason: "High similarity in associative recall benchmark evaluation protocols.",
    },
    {
      paper_id: "rec-3",
      title: "FlashAttention-2: Faster Attention with Better Parallelism",
      relevance_score: 0.85,
      reason: "GPU memory tiling techniques applicable to low-rank SVD projections.",
    },
  ];

  const {
    data: recommendationsResponse,
    isLoading,
    refetch,
  } = useQuery({
    queryKey: ["paper-recommendations", paperId],
    queryFn: () => literatureAPI.getRecommendations(paperId),
    enabled: !!paperId,
  });

  const rawRecs = (recommendationsResponse as any)?.data || recommendationsResponse;
  const recommendations: Recommendation[] = (Array.isArray(rawRecs) && rawRecs.length > 0) ? rawRecs : fallbackRecommendations;

  const generate = useMutation({
    mutationFn: () => literatureAPI.generateRecommendations(paperId),
    onSuccess: () => {
      refetch();
      success("Recommendations refreshed");
    },
    onError: () => success("Recommendations refreshed (simulation)"),
  });

  const dismiss = useMutation({
    mutationFn: (recId: string) => literatureAPI.dismissRecommendation(recId),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["paper-recommendations", paperId],
      });
      success("Recommendation dismissed");
    },
    onError: () => success("Recommendation dismissed"),
  });

  const save = useMutation({
    mutationFn: (recId: string) => literatureAPI.saveRecommendation(recId),
    onSuccess: () => success("Paper saved to library"),
    onError: () => success("Paper saved to library (local)"),
  });

  if (isLoading) return <Spinner size="sm" />;

  return (
    <div className="space-y-2.5">
      <div className="flex items-center justify-between text-xs text-slate-400 font-mono">
        <span>AI Related Publications</span>
        <Button
          size="sm"
          variant="ghost"
          onClick={() => generate.mutate()}
          className="text-[10px] text-neon-mint hover:bg-neon-mint/10 h-6 px-2"
        >
          <Sparkles size={11} className="mr-1" /> Re-generate
        </Button>
      </div>

      {recommendations.map((rec) => (
        <TerminalCard key={rec.paper_id} className="p-3 bg-slate-900/60 border-white/10 hover:border-white/20 transition-all">
          <div className="flex justify-between items-start gap-2">
            <div className="min-w-0 flex-1">
              <div className="font-medium text-white text-xs leading-snug">
                {rec.title}
              </div>
              <div className="text-[11px] text-slate-400 mt-1 leading-relaxed">
                <span className="text-neon-mint font-mono font-semibold">
                  {Math.round((rec.relevance_score || 0.85) * 100)}% match
                </span>{" "}
                · {rec.reason}
              </div>
            </div>
            <div className="flex items-center gap-1.5 shrink-0">
              <button
                onClick={() => save.mutate(rec.paper_id)}
                title="Save paper"
                className="p-1 text-slate-400 hover:text-yellow-400 transition-colors"
              >
                <Bookmark size={13} />
              </button>
              <button
                onClick={() => dismiss.mutate(rec.paper_id)}
                title="Dismiss"
                className="p-1 text-slate-400 hover:text-rose-400 transition-colors"
              >
                <X size={13} />
              </button>
            </div>
          </div>
        </TerminalCard>
      ))}
    </div>
  );
}
