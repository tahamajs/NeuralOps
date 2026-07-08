import { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { expertWallAPI } from "../services/api";
import { ApplicationWizard } from "../components/ApplicationWizard";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { ArrowLeft } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function ProjectApplicationPanel({
  onClose,
}: {
  onClose?: () => void;
}) {
  const { projectId } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const submitMutation = useMutation({
    mutationFn: (data: any) =>
      expertWallAPI.submitApplication(projectId!, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["applications"] });
      success("Application submitted", "The expert has been notified.");
      navigate("/dashboard");
    },
    onError: () => error("Submission failed", "Please try again."),
  });

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="sm" onClick={() => navigate(-1)}>
            <ArrowLeft size={16} />
          </Button>
          <h1 className="text-xl font-bold text-white">Apply to Project</h1>
        </div>
        {onClose && (
          <Button variant="ghost" size="sm" onClick={onClose}>
            ✕
          </Button>
        )}
      </div>
      <div className="flex-1 overflow-auto p-4">
        <ApplicationWizard
          onSubmit={(data) => submitMutation.mutate(data)}
          isSubmitting={submitMutation.isPending}
        />
      </div>
    </div>
  );
}
