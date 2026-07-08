import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { forensicsAPI } from "../services/api";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { Modal } from "../../../components/ui/Modal";
import { CheckCircle, XCircle, Play } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

interface FixApplicationPanelProps {
  analysis: any;
  open: boolean;
  onClose: () => void;
}

export function FixApplicationPanel({
  analysis,
  open,
  onClose,
}: FixApplicationPanelProps) {
  const queryClient = useQueryClient();
  const { success, error } = useToast();
  const { data: fixes, isLoading } = useQuery({
    queryKey: ["fixes", analysis.id],
    queryFn: () => forensicsAPI.getFixes(analysis.id),
    enabled: open && !!analysis.id,
  });

  const applyFix = useMutation({
    mutationFn: (fixId: string) => forensicsAPI.applyFix(fixId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["fixes", analysis.id] });
      success("Fix applied", "The suggested change has been executed.");
    },
    onError: () => error("Failed to apply fix"),
  });

  return (
    <Modal
      open={open}
      onClose={onClose}
      title={`Fix Failure: ${analysis.error_type}`}
    >
      <div className="space-y-4">
        <TerminalCard className="p-3 border-red-500/30">
          <div className="text-xs text-slate-400">Root Cause</div>
          <div className="text-white text-sm mt-1">{analysis.root_cause}</div>
        </TerminalCard>
        <TerminalCard className="p-3">
          <div className="text-xs text-slate-400">Suggested Fix</div>
          <div className="text-white text-sm mt-1">
            {analysis.suggested_fix}
          </div>
        </TerminalCard>
        {analysis.stack_trace && (
          <TerminalCard className="p-3">
            <div className="text-xs text-slate-400">Stack Trace</div>
            <pre className="text-xs text-red-300 mt-1 overflow-auto max-h-48">
              {analysis.stack_trace}
            </pre>
          </TerminalCard>
        )}
        <div className="space-y-2">
          <div className="text-sm font-bold text-white">Available Fixes</div>
          {isLoading ? (
            <Spinner size="sm" />
          ) : (
            fixes?.map((fix) => (
              <div
                key={fix.id}
                className="flex justify-between items-center p-3 bg-white/5 rounded"
              >
                <div>
                  <div className="text-sm text-white">{fix.description}</div>
                  <div className="text-xs text-slate-400 font-mono">
                    {fix.command}
                  </div>
                </div>
                <Button
                  size="sm"
                  onClick={() => applyFix.mutate(fix.id)}
                  disabled={fix.status !== "pending"}
                >
                  {fix.status === "pending" ? (
                    <Play size={14} />
                  ) : fix.status === "applied" ? (
                    <CheckCircle size={14} className="text-green-400" />
                  ) : (
                    <XCircle size={14} className="text-red-400" />
                  )}
                </Button>
              </div>
            ))
          )}
        </div>
      </div>
    </Modal>
  );
}
