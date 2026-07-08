import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { anomalyAPI } from "../services/api";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { CheckCircle, XCircle } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export function SuggestedFixes({ anomalyId }: { anomalyId: string }) {
  const queryClient = useQueryClient();
  const { success, error } = useToast();
  const { data: fixes, isLoading } = useQuery({
    queryKey: ["anomaly-fixes", anomalyId],
    queryFn: () => anomalyAPI.getFixes(anomalyId),
  });

  const applyFix = useMutation({
    mutationFn: (fixId: string) => anomalyAPI.applyFix(fixId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["anomaly-fixes", anomalyId] });
      success("Fix applied");
    },
    onError: () => error("Failed to apply fix"),
  });

  const fixesData = (fixes as any)?.data || fixes;

  if (isLoading) return <Spinner size="sm" />;
  if (!fixesData?.length)
    return (
      <div className="text-slate-500 text-sm">
        No automatic fixes available.
      </div>
    );

  return (
    <div className="space-y-3">
      {fixesData.map((fix: any) => (
        <TerminalCard key={fix.id} className="p-4">
          <div className="flex justify-between items-start">
            <div>
              <div className="font-medium text-white">{fix.description}</div>
              <div className="text-xs text-slate-400 mt-1">
                Command: <code className="text-neon-mint">{fix.command}</code>
              </div>
            </div>
            <Button
              size="sm"
              onClick={() => applyFix.mutate(fix.id)}
              disabled={applyFix.isPending || fix.status !== "pending"}
            >
              {fix.status === "pending" ? (
                "Apply"
              ) : fix.status === "applied" ? (
                <CheckCircle size={14} />
              ) : (
                <XCircle size={14} />
              )}
            </Button>
          </div>
        </TerminalCard>
      ))}
    </div>
  );
}
