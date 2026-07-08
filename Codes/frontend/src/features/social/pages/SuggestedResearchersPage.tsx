// src/features/social/pages/SuggestedResearchersPage.tsx
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { socialAPI } from "../services/api";
import { ResearcherCard } from "../components/ResearcherCard";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { Users, RefreshCw } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function SuggestedResearchersPage({
  onClose,
}: {
  onClose?: () => void;
}) {
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const {
    data: suggestions,
    isLoading,
    refetch,
  } = useQuery({
    queryKey: ["suggested-researchers"],
    queryFn: () => socialAPI.getSuggestedResearchers(),
  });

  const dismissSuggestion = useMutation({
    mutationFn: (id: string) => socialAPI.dismissSuggestion(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["suggested-researchers"] });
      success("Suggestion dismissed");
    },
  });

  if (isLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Users size={20} className="text-neon-mint" /> Suggested Researchers
          </h1>
          <p className="text-xs text-slate-400">
            Discover potential collaborators
          </p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant="ghost" onClick={() => refetch()}>
            <RefreshCw size={14} />
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4">
        {suggestions?.length ? (
          <div className="grid md:grid-cols-2 gap-4">
            {suggestions.map((s) => (
              <ResearcherCard
                key={s.id}
                researcher={s}
                onDismiss={() => dismissSuggestion.mutate(s.id)}
              />
            ))}
          </div>
        ) : (
          <div className="text-center text-slate-500 py-12">
            No suggestions at this time
          </div>
        )}
      </div>
    </div>
  );
}
